#ifndef __DG_TRIVIAL_SERIALIZER_H__
#define __DG_TRIVIAL_SERIALIZER_H__

#include <climits>
#include <bit>
#include <optional>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <tuple>

namespace dg::trivial_serializer::constants{

    static constexpr auto endianness = std::endian::little;
}

namespace dg::trivial_serializer::types{

    using size_type = uint64_t;
}

namespace dg::trivial_serializer::types_space{

    static constexpr auto nil_lambda = [](...){}; 

    template <class T, class = void>
    struct is_tuple: std::false_type{};
    
    template <class T>
    struct is_tuple<T, std::void_t<decltype(std::tuple_size<T>::value)>>: std::true_type{};

    template <class T>
    struct is_optional: std::false_type{};

    template <class ...Args>
    struct is_optional<std::optional<Args...>>: std::true_type{}; 

    template <class T, class = void>
    struct is_reflectible: std::false_type{};

    template <class T>
    struct is_reflectible<T, std::void_t<decltype(std::declval<T>().dg_reflect(nil_lambda))>>: std::true_type{};
    
    template <class T, class = void>
    struct is_dg_arithmetic: std::is_arithmetic<T>{};

    template <class T>
    struct is_dg_arithmetic<T, std::void_t<std::enable_if_t<std::is_floating_point_v<T>>>>: std::bool_constant<std::numeric_limits<T>::is_iec559>{}; 

    template <class T>
    static constexpr bool is_tuple_v        = is_tuple<T>::value; 

    template <class T>
    static constexpr bool is_optional_v     = is_optional<T>::value;

    template <class T>
    static constexpr bool is_reflectible_v  = is_reflectible<T>::value;

    template <class T>
    static constexpr bool is_dg_arithmetic_v = is_dg_arithmetic<T>::value;

    template <class T>
    struct base_type: std::enable_if<true, T>{};

    template <class T>
    struct base_type<const T>: base_type<T>{};

    template <class T>
    struct base_type<volatile T>: base_type<T>{};

    template <class T>
    struct base_type<T&>: base_type<T>{};

    template <class T>
    struct base_type<T&&>: base_type<T>{};

    template <class T>
    using base_type_t = typename base_type<T>::type;
}

namespace dg::trivial_serializer::utility{

    using namespace trivial_serializer::types;

    template <size_t N>
    static constexpr void memcpy(char * dst, const char * src, const std::integral_constant<size_t, N>){

        [=]<size_t ...IDX>(const std::index_sequence<IDX...>){
            ((dst[IDX] = src[IDX]), ...);
        }(std::make_index_sequence<N>());
    }  

    struct SyncedEndiannessService{
        
        static constexpr auto is_native_big      = bool{std::endian::native == std::endian::big};
        static constexpr auto is_native_little   = bool{std::endian::native == std::endian::little};
        static constexpr auto precond            = bool{(is_native_big ^ is_native_little) != 0};
        static constexpr auto deflt              = constants::endianness; 
        static constexpr auto native_uint8       = is_native_big ? uint8_t{0} : uint8_t{1}; 

        static_assert(precond); //xor

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static constexpr T bswap(T value){
            
            auto src = std::array<char, sizeof(T)>{};
            auto dst = std::array<char, sizeof(T)>{};
            const auto idx_seq  = std::make_index_sequence<sizeof(T)>();
            
            src = std::bit_cast<decltype(src)>(value);
            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                ((dst[IDX] = src[sizeof(T) - IDX - 1]), ...);
            }(idx_seq);
            value = std::bit_cast<T>(dst);

            return value;
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static constexpr void dump(char * dst, T data) noexcept{    

            if constexpr(std::endian::native != deflt){
                data = bswap(data);
            }

            std::array<char, sizeof(T)> data_buf{};
            data_buf = std::bit_cast<decltype(data_buf)>(data);
            memcpy(dst, data_buf.data(), std::integral_constant<size_t, sizeof(T)>());
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static constexpr T load(const char * src) noexcept{
            
            std::array<char, sizeof(T)> tmp{};
            memcpy(tmp.data(), src, std::integral_constant<size_t, sizeof(T)>());
            T rs = std::bit_cast<T>(tmp);

            if constexpr(std::endian::native != deflt){
                rs = bswap(rs);
            }

            return rs;
        }
    };
}

namespace dg::trivial_serializer::archive{
    
    struct IsSerializable{

        template <class T>
        constexpr auto is_serializable(T&& data) const noexcept -> bool{

            using btype = types_space::base_type_t<T>;
            
            if constexpr(types_space::is_dg_arithmetic_v<btype>){
                return true;
            } else if constexpr(types_space::is_optional_v<btype>){
                return is_serializable(data.value());
            } else if constexpr(types_space::is_tuple_v<btype>){
                return [&]<size_t ...IDX>(const std::index_sequence<IDX...>) noexcept{
                    return (IsSerializable{}.is_serializable(std::get<IDX>(data)) && ...);
                }(std::make_index_sequence<std::tuple_size_v<btype>>{});
            } else if constexpr(types_space::is_reflectible_v<btype>){
                bool rs = true;
                auto archiver = [&rs]<class ...Args>(Args&& ...args) noexcept{
                    rs &= (IsSerializable().is_serializable(std::forward<Args>(args)) && ...);
                };
                data.dg_reflect(archiver);
                return rs;
            } else{
                return false;
            }
        }
    };

    struct Counter{
        
        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            return sizeof(types_space::base_type_t<T>);
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            using value_type = typename types_space::base_type_t<T>::value_type;
            return count(bool{}) + count(value_type());
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            using btype         = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<btype>>{};

            return []<size_t ...IDX>(T&& data, const std::index_sequence<IDX...>) noexcept{
                return (Counter{}.count(std::get<IDX>(data)) + ...);
            }(std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr auto count(T&& data) const noexcept -> size_t{

            size_t rs{};
            auto archiver = [&rs]<class ...Args>(Args&& ...args) noexcept{
                rs += (Counter{}.count(std::forward<Args>(args)) + ...);
            };

            static_assert(noexcept(data.dg_reflect(archiver)));
            data.dg_reflect(archiver);

            return rs;
        }
    };

    struct Forward{

        using Self = Forward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using _MemIO = utility::SyncedEndiannessService;
            _MemIO::dump(buf, data);
            buf += sizeof(types_space::base_type_t<T>);
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using value_type = typename types_space::base_type_t<T>::value_type;
            char * tmp = buf;
            put(tmp, static_cast<bool>(data));

            if (data){
                put(tmp, data.value());
            }

            buf += Counter{}.count(data);
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            using btype = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<btype>>{};

            []<size_t ...IDX>(char *& buf, T&& data, const std::index_sequence<IDX...>) noexcept{
                (Self{}.put(buf, std::get<IDX>(data)), ...);
            }(buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(char *& buf, T&& data) const noexcept{

            auto archiver = [&buf]<class ...Args>(Args&& ...args) noexcept{
                (Self{}.put(buf, std::forward<Args>(args)), ...);
            };

            static_assert(noexcept(data.dg_reflect(archiver)));
            data.dg_reflect(archiver);
        }
    };

    struct Backward{

        using Self  = Backward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const noexcept{

            using btype     = types_space::base_type_t<T>;
            using _MemIO    = utility::SyncedEndiannessService;
            data            = _MemIO::load<btype>(buf);
            buf             += sizeof(btype);
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const noexcept{

            using obj_type  = std::remove_reference_t<decltype(*data)>;
            auto tmp        = buf;
            bool status     = {}; 
            put(tmp, status);

            if (status){
                auto obj = obj_type();
                put(tmp, obj);
                data = std::move(obj);
            } else{
                data = {};
            }

            buf += Counter{}.count(data);
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const noexcept{

            using btype         = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<btype>>{};

            []<size_t ...IDX>(const Self& _self, const char *& buf, T&& data, const std::index_sequence<IDX...>){
                (_self.put(buf, std::get<IDX>(data)), ...);
            }(*this, buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        constexpr void put(const char *& buf, T&& data) const noexcept{

            auto archiver   = [&buf]<class ...Args>(Args&& ...args){
                (Self().put(buf, std::forward<Args>(args)), ...);
            };

            static_assert(noexcept(data.dg_reflect(archiver)));
            data.dg_reflect(archiver);
        }
    };
}

namespace dg::trivial_serializer{

    template <class T, class = void>
    struct is_serializable: std::false_type{};

    template <class T>
    struct is_serializable<T, std::void_t<std::enable_if_t<std::is_trivial_v<T> && archive::IsSerializable().is_serializable(T{})>>>: std::true_type{};
    
    template <class T>
    static inline constexpr bool is_serializable_v = is_serializable<T>::value;
    
    template <class T>
    constexpr auto size(T&& obj) noexcept -> size_t{

        return archive::Counter{}.count(std::forward<T>(obj));
    }

    template <class T>
    constexpr auto serialize_into(char * buf, const T& obj) noexcept -> char *{

        archive::Forward().put(buf, obj);
        return buf;
    }

    template <class T>
    constexpr auto deserialize_into(T& obj, const char * buf) noexcept -> const char *{

        archive::Backward().put(buf, obj);
        return buf;
    } 
}

#endif
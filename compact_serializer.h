#ifndef __DG_COMPACT_SERIALIZER_H__
#define __DG_COMPACT_SERIALIZER_H__

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <climits>
#include <bit>
#include <optional>
#include <numeric>
#include "hasher.h"
#include <type_traits>
#include <array>

namespace dg::compact_serializer::constants{

    static constexpr auto endianness    = std::endian::little;
}

namespace dg::compact_serializer::types{

    using hash_type     = uint64_t; 
    using size_type     = uint64_t;
}

namespace dg::compact_serializer::types_space{

    static constexpr auto nil_lambda    = [](...){}; 

    template <class T, class = void>
    struct is_tuple: std::false_type{};
    
    template <class T>
    struct is_tuple<T, std::void_t<decltype(std::tuple_size<T>::value)>>: std::true_type{};

    template <class T>
    struct is_unique_ptr: std::false_type{};

    template <class T>
    struct is_unique_ptr<std::unique_ptr<T>>: std::bool_constant<!std::is_array_v<T>>{}; 

    template <class T>
    struct is_optional: std::false_type{};

    template <class ...Args>
    struct is_optional<std::optional<Args...>>: std::true_type{}; 

    template <class T>
    struct is_vector: std::false_type{};

    template <class ...Args>
    struct is_vector<std::vector<Args...>>: std::true_type{};

    template <class T>
    struct is_unordered_map: std::false_type{};

    template <class ...Args>
    struct is_unordered_map<std::unordered_map<Args...>>: std::true_type{}; 

    template <class T>
    struct is_unordered_set: std::false_type{};

    template <class ...Args>
    struct is_unordered_set<std::unordered_set<Args...>>: std::true_type{};

    template <class T>
    struct is_map: std::false_type{};

    template <class ...Args>
    struct is_map<std::map<Args...>>: std::true_type{}; 
    
    template <class T>
    struct is_set: std::false_type{};

    template <class ...Args>
    struct is_set<std::set<Args...>>: std::true_type{};

    template <class T>
    struct is_basic_string: std::false_type{};

    template <class ...Args>
    struct is_basic_string<std::basic_string<Args...>>: std::true_type{};

    template <class T, class = void>
    struct is_reflectible: std::false_type{};

    template <class T>
    struct is_reflectible<T, std::void_t<decltype(std::declval<T>().dg_reflect(nil_lambda))>>: std::true_type{};
    
    template <class T, class = void>
    struct is_dg_arithmetic: std::is_arithmetic<T>{};

    template <class T>
    struct is_dg_arithmetic<T, std::void_t<std::enable_if_t<std::is_floating_point_v<T>>>>: std::bool_constant<std::numeric_limits<T>::is_iec559>{}; 

    template <class T, class = void>
    struct container_value_or_empty{};

    template <class T>
    struct container_value_or_empty<T, std::void_t<typename T::value_type>>{
        using type = typename T::value_type;
    };

    template <class T, class = void>
    struct container_bucket_or_empty{};

    template <class T>
    struct container_bucket_or_empty<T, std::void_t<typename T::key_type, typename T::mapped_type>>{
        using type = std::pair<typename T::key_type, typename T::mapped_type>; //I dont want to complicate this further by adding const to key_type (since this is an application) - 
    };


    template <class T>
    using containee_or_empty = std::conditional_t<std::disjunction_v<is_vector<T>, is_unordered_set<T>, is_set<T>, is_basic_string<T>>,
                                                                     container_value_or_empty<T>,
                                                                     std::conditional_t<std::disjunction_v<is_unordered_map<T>, is_map<T>>, 
                                                                                        container_bucket_or_empty<T>, 
                                                                                        void>>;

    template <class T>
    using containee_t                           = typename containee_or_empty<T>::type;

    template <class T>
    static constexpr bool is_container_v        = std::disjunction_v<is_vector<T>, is_unordered_map<T>, is_unordered_set<T>, is_map<T>, is_set<T>, is_basic_string<T>>;

    template <class T>
    static constexpr bool is_tuple_v            = is_tuple<T>::value; 

    template <class T>
    static constexpr bool is_unique_ptr_v       = is_unique_ptr<T>::value;

    template <class T>
    static constexpr bool is_optional_v         = is_optional<T>::value;

    template <class T>
    static constexpr bool is_reflectible_v      = is_reflectible<T>::value;

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
    
    template <class T>
    static constexpr bool is_dg_arithmetic_v    = is_dg_arithmetic<T>::value;
}

namespace dg::compact_serializer::utility{

    using namespace compact_serializer::types;

    struct SyncedEndiannessService{
        
        static constexpr auto is_native_big      = bool{std::endian::native == std::endian::big};
        static constexpr auto is_native_little   = bool{std::endian::native == std::endian::little};
        static constexpr auto precond            = bool{(is_native_big ^ is_native_little) != 0};
        static constexpr auto deflt              = constants::endianness; 
        static constexpr auto native_uint8       = is_native_big ? uint8_t{0} : uint8_t{1}; 

        static_assert(precond); //xor

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static inline T bswap(T value) noexcept{
            
            std::array<char, sizeof(T)> src{};
            std::array<char, sizeof(T)> dst{};
            const auto idx_seq = std::make_index_sequence<sizeof(T)>();
            
            std::memcpy(src.data(), &value, sizeof(T));
            [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
                ((dst[IDX] = src[sizeof(T) - IDX - 1]), ...);
            }(idx_seq);
            std::memcpy(&value, dst.data(), sizeof(T));

            return value;
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static inline void dump(void * dst, T data) noexcept{    

            if constexpr(std::endian::native != deflt){
                data = bswap(data);
            }

            std::memcpy(dst, &data, sizeof(T));
        }

        template <class T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
        static inline T load(const void * src) noexcept{
            
            T rs{};
            std::memcpy(&rs, src, sizeof(T));

            if constexpr(std::endian::native != deflt){
                rs = bswap(rs);
            }

            return rs;
        }
    };

    auto hash(const char * buf, size_t sz) noexcept -> hash_type{
        
        static_assert(std::is_same_v<hash_type, size_t>); //stricter req for now
        return dg::hasher::hash_bytes(buf, sz);
    }

    template <class T, std::enable_if_t<std::disjunction_v<types_space::is_vector<T>, 
                                                           types_space::is_basic_string<T>>, bool> = true>
    constexpr auto get_inserter() noexcept{

        auto inserter   = []<class U, class K>(U&& container, K&& arg){
            container.push_back(std::forward<K>(arg));
        };

        return inserter;
    }

    template <class T, std::enable_if_t<std::disjunction_v<types_space::is_unordered_map<T>, 
                                                           types_space::is_unordered_set<T>, 
                                                           types_space::is_map<T>, 
                                                           types_space::is_set<T>>, bool> = true>
    constexpr auto get_inserter() noexcept{ 

        auto inserter   = []<class U, class K>(U&& container, K&& args){
            container.insert(std::forward<K>(args));
        };

        return inserter;
    }
}

namespace dg::compact_serializer::archive{

    struct Counter{
        
        using Self = Counter;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        auto count(T&& data) const noexcept -> size_t{
            
            return sizeof(types_space::base_type_t<T>);
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(bool{});

            if (data){
                rs += this->count(*data);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        auto count(T&& data) const noexcept -> size_t{

            size_t rs = this->count(bool{}); 

            if (data){
                rs += this->count(*data);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        auto count(T&& data) const noexcept -> size_t{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};
            size_t rs           = {};

            [&rs]<size_t ...IDX>(T&& data, const std::index_sequence<IDX...>) noexcept{
                rs += (Self().count(std::get<IDX>(data)) + ...);
            }(std::forward<T>(data), idx_seq);

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_container_v<types_space::base_type_t<T>>, bool> = true>
        auto count(T&& data) const noexcept -> size_t{
            
            size_t rs = this->count(types::size_type{});

            for (const auto& e: data){
                rs += this->count(e);
            }

            return rs;
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        auto count(T&& data) const noexcept -> size_t{

            size_t rs       = {};
            auto archiver   = [&rs]<class ...Args>(Args&& ...args) noexcept{
                rs += (Self().count(std::forward<Args>(args)) + ...);
            };
            data.dg_reflect(archiver);
            
            return rs;
        }
    };

    struct Forward{
        
        using Self = Forward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        void put(char *& buf, T&& data) const noexcept{
            
            utility::SyncedEndiannessService::dump(buf, std::forward<T>(data));
            std::advance(buf, sizeof(types_space::base_type_t<T>));
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        void put(char *& buf, T&& data) const noexcept{

            this->put(buf, static_cast<bool>(data));

            if (data){
                this->put(buf, *data);
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        void put(char *& buf, T&& data) const noexcept{

            this->put(buf, static_cast<bool>(data));

            if (data){
                this->put(buf, *data);
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        void put(char *& buf, T&& data) const noexcept{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            []<size_t ...IDX>(char *& buf, T&& data, const std::index_sequence<IDX...>) noexcept{
                (Self().put(buf, std::get<IDX>(data)), ...);
            }(buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_container_v<types_space::base_type_t<T>>, bool> = true>
        void put(char *& buf, T&& data) const noexcept{
            
            this->put(buf, static_cast<types::size_type>(data.size()));

            //optimizable - worth or not worth it
            for (const auto& e: data){
                this->put(buf, e);
            }
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        void put(char *& buf, T&& data) const noexcept{

            auto archiver = [&buf]<class ...Args>(Args&& ...args) noexcept{
                (Self().put(buf, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };

    struct Backward{

        using Self = Backward;

        template <class T, std::enable_if_t<types_space::is_dg_arithmetic_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, T&& data) const{

            using base_type = types_space::base_type_t<T>;
            data = utility::SyncedEndiannessService::load<base_type>(buf);
            std::advance(buf, sizeof(base_type));
        }

        template <class T, std::enable_if_t<types_space::is_unique_ptr_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, T&& data) const{

            using containee_type = typename types_space::base_type_t<T>::element_type;
            bool status = {};
            this->put(buf, status);

            if (status){
                auto obj = containee_type{};
                this->put(buf, obj);
                data = std::make_unique<containee_type>(std::move(obj));
            } else{
                data = nullptr;
            }
        }

        template <class T, std::enable_if_t<types_space::is_optional_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, T&& data) const{

            using containee_type = typename types_space::base_type_t<T>::value_type;
            bool status = {};
            this->put(buf, status);

            if (status){
                auto obj = containee_type{};
                this->put(buf, obj);
                data = std::optional<containee_type>(std::in_place_t{}, std::move(obj)); //fine
            } else{
                data = std::nullopt;
            }
        }

        template <class T, std::enable_if_t<types_space::is_tuple_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, T&& data) const{

            using base_type     = types_space::base_type_t<T>;
            const auto idx_seq  = std::make_index_sequence<std::tuple_size_v<base_type>>{};

            []<size_t ...IDX>(const char *& buf, T&& data, const std::index_sequence<IDX...>){
                (Self().put(buf, std::get<IDX>(data)), ...);
            }(buf, std::forward<T>(data), idx_seq);
        }

        template <class T, std::enable_if_t<types_space::is_container_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, T&& data) const{
            
            using base_type = types_space::base_type_t<T>;
            using elem_type = types_space::containee_t<base_type>;
            auto sz         = types::size_type{};
            auto isrter     = utility::get_inserter<base_type>();

            this->put(buf, sz); 
            data.reserve(sz);

            //optimizable - worth or not worth it
            for (size_t i = 0; i < sz; ++i){
                elem_type e{};
                this->put(buf, e);
                isrter(data, std::move(e));
            }
        }

        template <class T, std::enable_if_t<types_space::is_reflectible_v<types_space::base_type_t<T>>, bool> = true>
        void put(const char *& buf, T&& data) const{

            auto archiver = [&buf]<class ...Args>(Args&& ...args){
                (Self().put(buf, std::forward<Args>(args)), ...);
            };

            data.dg_reflect(archiver);
        }
    };
}

namespace dg::compact_serializer{

    //defined if: involving types c {std_arithmetic, std::tuple and friends, std::vector, std::unordered_map, std::map, std::unrodered_set, std::set, std::optional, std::basic_string, std::unique_ptr, dg_reflectible}
    //            involving types - except the ones coerced by internal functions - are base types (no const no reference) 
    
    //a class is dg_reflectible qualified if (1): it's members are dg_reflectible-qualfied - refer to involving types
    //                                       (2): implements reflectible: by defining two same-functionality-different-signature public methods - dg_reflect(const Reflector&) const and dg_reflect(const Reflector&)
    //                                       (3): does not meets other requirements (std::tuple and friends, std::vector<>, ...)
    //                                       (4): is the final and sole component that implements reflectible

    //undefined if not in defined

    struct bad_encoding_format: std::exception{}; 

    template <class T>
    auto size(const T& obj) noexcept -> size_t{

        return archive::Counter{}.count(obj);
    }

    template <class T>
    auto serialize_into(char * buf, const T& obj) noexcept -> char *{

        archive::Forward{}.put(buf, obj);
        return buf;
    } 

    template <class T>
    auto deserialize_into(T& obj, const char * buf) -> const char *{

        archive::Backward().put(buf, obj);
        return buf;
    }

    template <class T>
    auto integrity_size(const T& obj) noexcept -> size_t{

        return size(obj) + size(types::hash_type{});
    }

    template <class T>
    auto integrity_serialize_into(char * buf, const T& obj) noexcept -> char *{ 
        
        char * first                = buf;
        char * last                 = serialize_into(first, obj);
        types::hash_type hashed     = utility::hash(first, std::distance(first, last));
        char * llast                = serialize_into(last, hashed);

        return llast;
    }

    template <class T>
    void integrity_deserialize_into(T& obj, const char * buf, size_t sz){

        if (sz < size(types::hash_type{})){
            throw bad_encoding_format();
        }

        const char * first          = buf;
        const char * last           = first + (sz - size(types::hash_type{})); 
        types::hash_type expected   = {};
        types::hash_type reality    = utility::hash(first, std::distance(first, last));
        deserialize_into(expected, last);

        if (expected != reality){
            throw bad_encoding_format();
        }

        deserialize_into(obj, first);
    }
}

#endif
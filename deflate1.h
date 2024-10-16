#ifndef __DG_DEFLATE_H__
#define __DG_DEFLATE_H__

#include <stdint.h>
#include <stdlib.h>
#include <array>
#include <optional>
#include <vector>
#include <string>
#include "dense_hash_map/dense_hash_map.hpp"
#include <utility>
#include "trivial_serializer.h"
#include "compact_serializer.h"
#include <optional>
#include "huffman_encoder.h"
#include "hasher.h"
#include <limits.h>

namespace dg::deflate1::sliding_window_encoder{

    template <class T, std::enable_if_t<dg::trivial_serializer::is_serializable_v<T>, bool> = true>
    struct trivial_reflectable_hasher{

        constexpr auto operator()(const T& value) const noexcept -> size_t{

            constexpr size_t SZ         = dg::trivial_serializer::size(T{});  
            std::array<char, SZ> buf    = {};
            dg::trivial_serializer::serialize_into(buf.data(), value);

            return dg::hasher::hash_bytes(buf.data(), std::integral_constant<size_t, SZ>{});
        }
    };

    static inline constexpr size_t MAX_BACKMAP_SIZE         = size_t{1} << 20;

    template <size_t BACKREFERENCE_SZ>
    using back_reference_t = std::conditional_t<BACKREFERENCE_SZ == 1u,
                                                uint8_t,
                                                std::conditional_t<BACKREFERENCE_SZ == 2u,
                                                                   uint16_t,
                                                                   void>>; 

    template <size_t BACKREFERENCE_SZ>
    struct BackTrackToken{
        back_reference_t<BACKREFERENCE_SZ> delta;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(delta);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(delta);
        }
    };

    template <size_t MAX_SLIDING_WINDOW_SZ>
    struct NormalToken{
        std::array<char, MAX_SLIDING_WINDOW_SZ> data;
    
        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(data);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(data);
        }
    };

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    struct Token{
        std::optional<BackTrackToken<BACKREFERENCE_SZ>> backtrack_token;
        std::optional<NormalToken<MAX_SLIDING_WINDOW_SZ>> normal_token;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(backtrack_token, normal_token);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(backtrack_token, normal_token);
        }    
    };

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    struct EncodedData{
        std::vector<Token<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>> token_vec;
        std::string rem;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(token_vec, rem);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(token_vec, rem);
        }
    };

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    struct EncodedDataX{
        std::vector<uint8_t> header_vec;
        std::vector<BackTrackToken<BACKREFERENCE_SZ>> back_track_vec;
        std::vector<NormalToken<MAX_SLIDING_WINDOW_SZ>> normal_token_vec;
        EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ> rem;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(header_vec, back_track_vec, normal_token_vec, rem);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(header_vec, back_track_vec, normal_token_vec, rem);
        }
    };

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    auto encode(const char * src, size_t src_sz, const std::integral_constant<size_t, MAX_SLIDING_WINDOW_SZ>, const std::integral_constant<size_t, BACKREFERENCE_SZ>) -> EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{

        static_assert(MAX_SLIDING_WINDOW_SZ != 0u);

        auto encoded            = EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{};
        auto back_map           = jg::dense_hash_map<std::array<char, MAX_SLIDING_WINDOW_SZ>, size_t, trivial_reflectable_hasher<std::array<char, MAX_SLIDING_WINDOW_SZ>>>();
        size_t tokenable_sz     = src_sz / MAX_SLIDING_WINDOW_SZ; 

        for (size_t i = 0u; i < tokenable_sz; ++i){
            const char * first  = src + (i * MAX_SLIDING_WINDOW_SZ);
            const char * last   = src + ((i + 1) * MAX_SLIDING_WINDOW_SZ);
            auto token          = std::array<char, MAX_SLIDING_WINDOW_SZ>{};
            std::copy(first, last, token.begin());
            auto map_ptr        = back_map.find(token);

            if (map_ptr == back_map.end()){
                NormalToken<MAX_SLIDING_WINDOW_SZ> appendee{};
                std::copy(first, last, appendee.data.begin());
                encoded.token_vec.push_back(Token<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{std::nullopt, std::move(appendee)});
            } else{
                size_t last_idx = map_ptr->second; 
                intmax_t max_acceptable_idx = static_cast<intmax_t>(i * MAX_SLIDING_WINDOW_SZ) - intmax_t{std::numeric_limits<back_reference_t<BACKREFERENCE_SZ>>::max()};

                if (static_cast<intmax_t>(last_idx) >= max_acceptable_idx){
                    back_reference_t<BACKREFERENCE_SZ> delta = i * MAX_SLIDING_WINDOW_SZ - last_idx;
                    BackTrackToken<BACKREFERENCE_SZ> appendee{delta};
                    encoded.token_vec.push_back(Token<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{std::move(appendee), std::nullopt});
                } else{
                    NormalToken<MAX_SLIDING_WINDOW_SZ> appendee{};
                    std::copy(first, last, appendee.data.begin());
                    encoded.token_vec.push_back(Token<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{std::nullopt, std::move(appendee)});
                    back_map.erase(map_ptr);
                }
            }

            if (i != 0u){
                for (size_t j = 0u; j < MAX_SLIDING_WINDOW_SZ; ++j){
                    const char * bfirst = src + (i * MAX_SLIDING_WINDOW_SZ - j);
                    const char * blast  = src + ((i + 1) * MAX_SLIDING_WINDOW_SZ - j); 
                    std::copy(bfirst, blast, token.begin());
                    back_map[token] = i * MAX_SLIDING_WINDOW_SZ - j;
                }
            }
        }
        
        std::copy(src + (tokenable_sz * MAX_SLIDING_WINDOW_SZ), src + src_sz, std::back_inserter(encoded.rem));        
        return encoded;
    }

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    auto decode(char * dst, EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ> data) -> char *{

        static_assert(MAX_SLIDING_WINDOW_SZ != 0u);
        char * op = dst;

        for (size_t i = 0u; i < data.token_vec.size(); ++i){
            if (data.token_vec[i].backtrack_token.has_value()){
                back_reference_t<BACKREFERENCE_SZ> delta = data.token_vec[i].backtrack_token->delta; 
                char * src = dst + (i * MAX_SLIDING_WINDOW_SZ - delta);
                op = std::copy(src, src + MAX_SLIDING_WINDOW_SZ, op);
            } else if (data.token_vec[i].normal_token.has_value()){
                op = std::copy(data.token_vec[i].normal_token->data.begin(), data.token_vec[i].normal_token->data.end(), op);
            } else{
                std::abort();
            }
        }

        return std::copy(data.rem.begin(), data.rem.end(), op);
    }

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    auto encode_x(EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ> data) -> EncodedDataX<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{

        auto rs = EncodedDataX<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{};
        size_t header_sz = data.token_vec.size() / CHAR_BIT; 

        for (size_t i = 0u; i < header_sz; ++i){
            uint8_t header = {};

            for (size_t j = 0u; j < CHAR_BIT; ++j){
                header <<= 1;
                header |= static_cast<uint8_t>(static_cast<bool>(data.token_vec[i * CHAR_BIT + j].backtrack_token));
            }

            rs.header_vec.push_back(header);
        }

        for (size_t i = 0u; i < data.token_vec.size(); ++i){
            if (data.token_vec[i].backtrack_token){
                rs.back_track_vec.push_back(*data.token_vec[i].backtrack_token);
            } else{
                rs.normal_token_vec.push_back(*data.token_vec[i].normal_token);
            }
        }

        auto first  = data.token_vec.begin();
        auto last   = data.token_vec.begin() + (header_sz * CHAR_BIT);
        data.token_vec.erase(first, last);
        
        rs.rem = std::move(data);
        return rs;
    }

    template <size_t MAX_SLIDING_WINDOW_SZ, size_t BACKREFERENCE_SZ>
    auto decode_x(EncodedDataX<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ> inp) -> EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{

        EncodedData<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ> rs{};
        size_t bitarr_sz        = inp.header_vec.size() * CHAR_BIT;
        size_t backtrack_off    = 0u;
        size_t normal_off       = 0u;

        for (size_t i = 0u; i < bitarr_sz; ++i){
            size_t slot = i / CHAR_BIT;
            size_t offs = i % CHAR_BIT;
            const uint8_t TOGGLE = uint8_t{1u} << (CHAR_BIT - offs - 1);
            bool flag   = (inp.header_vec[slot] & TOGGLE) != 0u;

            if (flag){
                rs.token_vec.push_back(Token<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{inp.back_track_vec[backtrack_off] ,std::nullopt});
                backtrack_off += 1;
            } else{
                rs.token_vec.push_back(Token<MAX_SLIDING_WINDOW_SZ, BACKREFERENCE_SZ>{std::nullopt, inp.normal_token_vec[normal_off]});
                normal_off += 1;
            }
        }
        
        rs.token_vec.insert(rs.token_vec.end(), inp.rem.token_vec.begin(), inp.rem.token_vec.end());
        rs.rem = inp.rem.rem;

        return rs;
    }
} 

namespace dg::deflate1{

    //this is fine for first cut proof of concept - a significant effort should be considered to actually be usable in practice 

    template <class T>
    auto serialize(const T& data) -> std::string{

        std::string bstream(dg::compact_serializer::size(data), ' ');
        dg::compact_serializer::serialize_into(bstream.data(), data);

        return bstream;
    }

    template <class T>
    auto deserialize(const std::string& bstream) -> T{

        T rs{};
        dg::compact_serializer::deserialize_into(rs, bstream.data());

        return rs;
    }

    struct DeflateMsg{
        uint64_t data_sz_0;
        uint64_t data_sz_1;
        std::string bstream_0;
        std::string bstream_1;
        std::string bstream_2;
        std::string bstream_3;
        std::string bstream_4;
        std::string bstream_5;
        std::string bstream_6;

        template <class Reflector>
        void dg_reflect(const Reflector& reflector) const{
            reflector(data_sz_0, data_sz_1, bstream_0, bstream_1, bstream_2, bstream_3, bstream_4, bstream_5, bstream_6);
        }

        template <class Reflector>
        void dg_reflect(const Reflector& reflector){
            reflector(data_sz_0, data_sz_1, bstream_0, bstream_1, bstream_2, bstream_3, bstream_4, bstream_5, bstream_6);
        }
    };

    auto encode(std::string data) -> std::string{

        auto encoded        = sliding_window_encoder::encode_x(sliding_window_encoder::encode(data.data(), data.size(), std::integral_constant<size_t, 8>{}, std::integral_constant<size_t, 2>{})); 
        auto bstream        = dg::huffman_encoder::encode(serialize(encoded.header_vec));
        auto bstream1       = dg::huffman_encoder::encode(serialize(encoded.back_track_vec));
        auto bstream2       = dg::huffman_encoder::encode(serialize(encoded.rem));
        auto bstream3       = serialize(encoded.normal_token_vec);
        auto _encoded       = sliding_window_encoder::encode_x(sliding_window_encoder::encode(bstream3.data(), bstream3.size(), std::integral_constant<size_t, 4>{}, std::integral_constant<size_t, 2>{})); 
        auto _bstream       = dg::huffman_encoder::encode(serialize(_encoded.header_vec));
        auto _bstream1      = dg::huffman_encoder::encode(serialize(_encoded.back_track_vec));
        auto _bstream2      = dg::huffman_encoder::encode(serialize(_encoded.normal_token_vec));
        auto _bstream3      = serialize(_encoded.rem);
        size_t data_sz      = data.size();
        size_t bstream3_sz  = bstream3.size(); 

        return serialize(DeflateMsg(data_sz, bstream3_sz, std::move(bstream), std::move(bstream1), std::move(bstream2), std::move(_bstream), std::move(_bstream1), std::move(_bstream2), std::move(_bstream3)));
    }

    auto decode(const std::string& data) -> std::string{

        DeflateMsg msg          = deserialize<DeflateMsg>(data);
        auto header1_vec        = deserialize<std::vector<uint8_t>>(dg::huffman_encoder::decode(msg.bstream_0));
        auto backtrack1_vec     = deserialize<std::vector<sliding_window_encoder::BackTrackToken<2>>>(dg::huffman_encoder::decode(msg.bstream_1));
        auto rem1               = deserialize<sliding_window_encoder::EncodedData<8, 2>>(dg::huffman_encoder::decode(msg.bstream_2));
        auto header2_vec        = deserialize<std::vector<uint8_t>>(dg::huffman_encoder::decode(msg.bstream_3));
        auto backtrack2_vec     = deserialize<std::vector<sliding_window_encoder::BackTrackToken<2>>>(dg::huffman_encoder::decode(msg.bstream_4));
        auto normaltoken2_vec   = deserialize<std::vector<sliding_window_encoder::NormalToken<4>>>(dg::huffman_encoder::decode(msg.bstream_5));
        auto rem2               = deserialize<sliding_window_encoder::EncodedData<4, 2>>(msg.bstream_6);

        sliding_window_encoder::EncodedDataX<4, 2> encoded_data{std::move(header2_vec), std::move(backtrack2_vec), std::move(normaltoken2_vec), std::move(rem2)};
        auto decoded            = sliding_window_encoder::decode_x(encoded_data);
        std::string bstream     = std::string(msg.data_sz_1, ' ');
        char * last             = sliding_window_encoder::decode(bstream.data(), decoded); 
        bstream.resize(std::distance(bstream.data(), last));

        auto normaltoken1_vec   = deserialize<std::vector<sliding_window_encoder::NormalToken<8>>>(bstream);
        sliding_window_encoder::EncodedDataX<8, 2> encoded_data_2{std::move(header1_vec), std::move(backtrack1_vec), std::move(normaltoken1_vec), std::move(rem1)};
        auto decoded2           = sliding_window_encoder::decode_x(encoded_data_2);
        std::string bstream2    = std::string(msg.data_sz_0, ' ');
        char * last2            = sliding_window_encoder::decode(bstream2.data(), decoded2);
        bstream2.resize(std::distance(bstream2.data(), last2));

        return bstream2;
    }
} 

#endif
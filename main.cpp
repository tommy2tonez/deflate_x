#include "deflate1.h"
#include <iostream>
#include <fstream>
#include <random>

int main(){

    // std::string msg = "test2";
    // std::string msgg = dg::huffman_encoder::decode(dg::huffman_encoder::encode(msg));
    // std::cout << msg << std::endl
    // ;
    // std::cout << msgg;
    // auto encoded    = std::string(dg::huffman_encoder::constants::MAX_ENCODING_SZ_PER_BYTE * s.size(), ' ');
    // char * last     = dg::huffman_encoder::encode_into(encoded.data(), s.data(), s.size(), model.get());
    // encoded.resize(std::distance(encoded.data(), last));

    // dg::huffman_encoder::decode_into(s.data(), encoded.data(), encoded.size(), model.get());

    // std::cout << s;

    // std::string msg = "tommy23";
    // std::string decoded = dg::deflate1::decode(dg::deflate1::encode(msg));
    // std::cout << decoded << std::endl;
    std::ifstream f_in("test.txt");
    f_in.seekg(0, f_in.end);
    size_t fsz = f_in.tellg();
    std::unique_ptr<char[]> buf = std::make_unique<char[]>(fsz);
    f_in.seekg(0, f_in.beg);
    f_in.read(buf.get(), fsz);

    std::string msg(buf.get(), buf.get() + fsz); 
    std::string encoded = dg::deflate1::encode(msg);
    std::string decoded = dg::deflate1::decode(encoded);
    std::cout << msg.size() << "<>" << encoded.size() << "<>" << (double{encoded.size()} / double{msg.size()})  << "<>" << (decoded == msg) << std::endl; 

    auto byte_rand_device   = std::bind(std::uniform_int_distribution<char>{}, std::mt19937{});
    auto sz_rand_device     = std::bind(std::uniform_int_distribution<uint8_t>{}, std::mt19937{});

    while (true){
        std::string test_data(sz_rand_device(), ' ');
        std::generate(test_data.begin(), test_data.end(), std::ref(byte_rand_device));
        std::string encoded_test_data   = dg::deflate1::encode(test_data);
        std::string decoded_test_data   = dg::deflate1::decode(encoded_test_data);

        if (decoded_test_data != test_data){
            std::cout << "mayday" << std::endl;
        }
    }

}
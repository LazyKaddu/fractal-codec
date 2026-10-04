#include <CLI/CLI.hpp>
#include <fractal/core/VideoEncoder.h>
#include <fractal/core/VideoDecoder.h>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"Fractal Video Codec CLI"};
    app.require_subcommand(1); // Enforce choosing either 'encode' or 'decode'

    // ==========================================
    // ENCODE COMMAND
    // ==========================================
    auto* encode_cmd = app.add_subcommand("encode", "Encode an mp4 video into a .frc fractal binary");
    
    std::string enc_input, enc_output;
    encode_cmd->add_option("-i,--input", enc_input, "Input video file (e.g. input.mp4)")
        ->required()
        ->check(CLI::ExistingFile);
        
    encode_cmd->add_option("-o,--output", enc_output, "Output compressed file (e.g. compressed.frc)")
        ->required();

    encode_cmd->callback([&]() {
        std::cout << "========================================\n";
        std::cout << "       FRACTAL VIDEO ENCODER GPU        \n";
        std::cout << "========================================\n";
        
        fractal::core::VideoEncoder encoder(enc_input, enc_output);
        if (!encoder.run()) {
            std::cerr << "Fatal Error: Encoding pipeline failed.\n";
            exit(1);
        }
    });

    // ==========================================
    // DECODE COMMAND
    // ==========================================
    auto* decode_cmd = app.add_subcommand("decode", "Decode a .frc fractal binary back into an mp4 video");
    
    std::string dec_input, dec_output;
    decode_cmd->add_option("-i,--input", dec_input, "Input compressed file (e.g. compressed.frc)")
        ->required()
        ->check(CLI::ExistingFile);
        
    decode_cmd->add_option("-o,--output", dec_output, "Output video file (e.g. decoded.mp4)")
        ->required();

    decode_cmd->callback([&]() {
        std::cout << "========================================\n";
        std::cout << "       FRACTAL VIDEO DECODER GPU        \n";
        std::cout << "========================================\n";
        
        fractal::core::VideoDecoder decoder(dec_input, dec_output);
        if (!decoder.run()) {
            std::cerr << "Fatal Error: Decoding pipeline failed.\n";
            exit(1);
        }
    });

    // This single macro handles parsing, printing help messages, and triggering the callbacks!
    CLI11_PARSE(app, argc, argv);
    
    return 0;
}

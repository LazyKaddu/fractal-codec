# Command Line Interface

The Fractal Codec uses the `CLI11` library to provide a modern, robust command-line experience.

## Building the CLI
To build the CLI executable, make sure you have CMake, a CUDA toolkit, OpenCV, and Zstd installed, then run:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Encoding a Video
To compress a standard video into the fractal `.frc` format, use the `encode` subcommand:
```bash
./fractal_cli encode -i my_video.mp4 -o compressed.frc
```

## Decoding a Video
To decompress a `.frc` file back into a playable video, use the `decode` subcommand:
```bash
./fractal_cli decode -i compressed.frc -o decoded_output.mp4
```

## Help Menu
If you ever need to check the available commands or see required arguments, you can view the auto-generated help menu by running:
```bash
./fractal_cli --help
```

cd "$(dirname "$(realpath "$0")")"

/usr/lib/emscripten/emcmake cmake \
    -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja

cmake --build build

ln -srf build/pieces.{js,wasm} website/public/

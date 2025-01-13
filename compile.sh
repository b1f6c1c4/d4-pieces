/usr/lib/emscripten/emcmake cmake \
    -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS=' -s ENVIRONMENT=web -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s EXPORT_NAME=Pieces -s WASM_BIGINT' -G Ninja

cmake --build build

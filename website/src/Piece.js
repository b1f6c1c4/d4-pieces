import React from 'react';

function hlsGen(seed) {
  if (isNaN(seed)) {
    seed = 0;
  }
  const random = mulberry32(seed);

  let preH = 0;
  function getH() {
    while (true) {
      const newH = random() * 360;
      if (Math.abs(preH - newH) > 10) {
        preH = newH;
        return newH;
      }
    }
  }

  return function() {
    const H = getH();
    const L = (40 + random() * 20) + "%";
    return `hsl(${H}, 100%, ${L})`;
  };
}

function mulberry32(seed = Date.now()) {
  return function() {
    let x = seed += 0x6D2B79F5;
    x = Math.imul(x ^ x >>> 15, x | 1);
    x ^= x + Math.imul(x ^ x >>> 7, x | 61);
    return ((x ^ x >>> 14) >>> 0) / 4294967296;
  }
}

const nextHls = hlsGen(114514);
const colors = Array(1000).fill(0).map(nextHls);

const Piece = ({ shapeId, stepId, shape, reduced, transform }) => {
  const bits = shape.value.toString(2).padStart(shape.LEN ** 2, '0').split('').reverse();

  const N = Math.max(shape.width, shape.height);
  const gridStyle = reduced ? {
    display: 'grid',
    gridTemplateColumns: `repeat(${N}, 20px)`,
    gridTemplateRows: `repeat(${N}, 20px)`,
    gap: '0',
    transition: 'transform 200ms ease-out 20ms',
    transform,
  } : {
    display: 'grid',
    gridTemplateColumns: `repeat(${shape.LEN}, 30px)`,
    gridTemplateRows: `repeat(${shape.LEN}, 30px)`,
    gap: '0',
  };
  if (stepId !== undefined) {
    const dur = 0.9;
    const del = 0.1 * stepId;
    gridStyle.animation = `${dur}s ease-out ${del}s 1 normal backwards slide-in`;
  }

  return (
    <div className="grid" style={gridStyle}>
      {bits.map((bit, index) => {
        const row = Math.floor(index / shape.LEN);
        const col = index % shape.LEN;
        if (reduced && row >= shape.height) return undefined;
        if (reduced && col >= shape.width) return undefined;
        const style = {
          gridColumn: col + 1,
          gridRow: row + 1,
        };
        if (bit === '0')
          return (
            <div
              key={index}
              className="tile"
              style={style}
            />
          );
        style.backgroundColor = colors[shapeId];
        const west = col > 0 && bits[index - 1] === '1';
        const east = col < shape.LEN - 1 && bits[index + 1] === '1';
        const north = row > 0 && bits[index - shape.LEN] === '1';
        const south = row < shape.LEN - 1 && bits[index + shape.LEN] === '1';
        style.borderLeftWidth = west ? '0' : '1.5px';
        style.borderRightWidth = east ? '0' : '1.5px';
        style.borderTopWidth = north ? '0' : '1.5px';
        style.borderBottomWidth = south ? '0' : '1.5px';
        style.borderRadius = [
          (!north && !west) ? '8px' : '0',
          (!north && !east) ? '8px' : '0',
          (!south && !east) ? '8px' : '0',
          (!south && !west) ? '8px' : '0',
        ].join(' ');
        return (
          <div
            key={index}
            className="tile"
            style={style}
          />
        );
      })}
    </div>
  );
};

export default Piece;

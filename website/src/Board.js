import React from 'react';

const Board = ({ shape, onToggle }) => {
  const { value, LEN } = shape;

  const style = {
    display: 'grid',
    gridTemplateColumns: `repeat(${shape.LEN}, 30px)`,
    gridTemplateRows: `repeat(${shape.LEN}, 30px)`,
    gap: '0',
  };

  // Helper to check if the bit at (row, col) is 1 or 0
  const isTileActive = (row, col) => {
    const bitIndex = window.BigInt(row * LEN + col);
    return (value >> bitIndex) & 1n;
  };

  // Render the grid
  const renderGrid = () => {
    const grid = [];
    for (let row = 0; row < LEN; row++) {
      for (let col = 0; col < LEN; col++) {
        const active = isTileActive(row, col);
        grid.push(
          <div
            key={`${row}-${col}`}
            onClick={() => onToggle && onToggle(row, col)}
            style={{
              backgroundColor: active ? 'grey' : 'black',
            }}
          />
        );
      }
    }
    return grid;
  };

  return <div className="board" style={style}>{renderGrid()}</div>;
};

export default Board;


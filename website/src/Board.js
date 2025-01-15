import React from 'react';

const Board = ({ shape, onToggle }) => {
  const { value, LEN } = shape;

  const style = {
    display: 'grid',
    gridTemplateColumns: `5px repeat(${shape.LEN}, 30px) 5px`,
    gridTemplateRows: `5px repeat(${shape.LEN}, 30px) 5px`,
    gap: '0',
  };

  // Helper to check if the bit at (row, col) is 1 or 0
  const isTileActive = (row, col) => {
    if (row < 0) return false;
    if (row >= shape.LEN) return false;
    if (col < 0) return false;
    if (col >= shape.LEN) return false;
    const bitIndex = window.BigInt(row * LEN + col);
    return !!((value >> bitIndex) & 1n);
  };

  // Render the grid
  const renderGrid = (extra) => {
    const grid = [];
    for (let row = -1; row <= LEN; row++) {
      for (let col = -1; col <= LEN; col++) {
        if (row < 0 || row >= shape.LEN || col < 0 || col >= shape.LEN) {
          grid.push(
            <div key={`${row}-${col}`} className="padding" />
          );
          continue;
        }
        const active = isTileActive(row, col);
        const west = isTileActive(row, col - 1);
        const east = isTileActive(row, col + 1);
        const north = isTileActive(row - 1, col);
        const south = isTileActive(row + 1, col);
        const nw = isTileActive(row - 1, col - 1);
        const ne = isTileActive(row - 1, col + 1);
        const sw = isTileActive(row + 1, col - 1);
        const se = isTileActive(row + 1, col + 1);
        if (extra) {
          grid.push(
            <div
              key={`${row}-${col}`}
              style={{
                backgroundColor: !active ? 'var(--tile)' : 'var(--dark-wood)',
              }}
            />
          );
          continue;
        }
        const style = { };
        style.borderLeftWidth   = !active ? west  ? '1px' : '0' : west  ? '1px' : '1px';
        style.borderRightWidth  = !active ? east  ? '1px' : '0' : east  ? '1px' : '1px';
        style.borderTopWidth    = !active ? north ? '1px' : '0' : north ? '1px' : '1px';
        style.borderBottomWidth = !active ? south ? '1px' : '0' : south ? '1px' : '1px';
        style.borderLeftColor   = active ? 'var(--tile-slit)' : west  ? 'var(--dark-wood)' : 'unset';
        style.borderRightColor  = active ? 'var(--tile-slit)' : east  ? 'var(--dark-wood)' : 'unset';
        style.borderTopColor    = active ? 'var(--tile-slit)' : north ? 'var(--dark-wood)' : 'unset';
        style.borderBottomColor = active ? 'var(--tile-slit)' : south ? 'var(--dark-wood)' : 'unset';
        style.borderRadius = [
          (active !== north && active !== west && (!active || !nw)) ? '8px' : '0',
          (active !== north && active !== east && (!active || !ne)) ? '8px' : '0',
          (active !== south && active !== east && (!active || !se)) ? '8px' : '0',
          (active !== south && active !== west && (!active || !sw)) ? '8px' : '0',
        ].join(' ');
        grid.push(
          <div
            key={`${row}-${col}`}
            onClick={() => onToggle && onToggle(row, col)}
            className={active ? 'active' : 'inactive'}
            style={style}
          />
        );
      }
    }
    return grid;
  };

  return <div className="board">
    <div style={style}>{renderGrid(true)}</div>
    <div style={style}>{renderGrid(false)}</div>
  </div>
};

export default Board;


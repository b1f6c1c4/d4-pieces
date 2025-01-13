import React from 'react';
import Piece from './Piece';

const PieceSelector = ({ shapeId, shape, avail, onAvailChange }) => {
  const handleDecrease = () => {
    if (avail > 0) onAvailChange(avail - 1);
  };

  const handleIncrease = () => {
    onAvailChange(avail + 1);
  };

  const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '10px'
  };

  const inputContainerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '5px'
  };

  return (
    <div style={containerStyle}>
      <Piece shape={shape} reduced shapeId={shapeId} />
      <div style={inputContainerStyle}>
        <button onClick={handleDecrease}>-</button>
        <input
          type="number"
          value={avail}
          onChange={(e) => onAvailChange(Number(e.target.value))}
          style={{ width: '50px', textAlign: 'center' }}
        />
        <button onClick={handleIncrease}>+</button>
      </div>
    </div>
  );
};

export default PieceSelector;

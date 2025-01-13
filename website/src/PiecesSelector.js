import React from 'react';
import PieceSelector from './PieceSelector';

const containerStyle = {
  display: 'flex',
  overflowX: 'auto',
  gap: '10px',
  padding: '10px',
};

const PiecesSelector = ({ shapes, onAvailChange }) => {
  return (
    <div style={containerStyle}>
      {shapes.map((item, index) => (
        <PieceSelector
          key={index}
          shapeId={index}
          shape={item.shape}
          avail={item.count}
          onAvailChange={(value) => onAvailChange(index, value)}
        />
      ))}
    </div>
  );
};

export default PiecesSelector;


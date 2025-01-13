import React from 'react';
import PieceSelector from './PieceSelector';

const containerStyle = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: '10px',
  padding: '10px',
};

const PiecesSelector = ({ module, pieces, onChange }) => {
  return (
    <div style={containerStyle}>
      {pieces.map((item, index) => (
        <PieceSelector
          key={index}
          module={module}
          shapeId={index}
          piece={item}
          onChange={onChange}
        />
      ))}
    </div>
  );
};

export default PiecesSelector;


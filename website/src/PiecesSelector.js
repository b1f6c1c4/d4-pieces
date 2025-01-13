import React from 'react';
import PieceSelector from './PieceSelector';

const PiecesSelector = ({ module, pieces, onChange }) => {
  return (
    <div className="pieces-selector">
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


import React from 'react';
import PieceSelector from './PieceSelector';

const PiecesSelector = ({ module, pieces, onUpdate }) => {
  return (
    <div className="pieces-selector">
      {pieces.map((item, index) => (
        <PieceSelector
          key={index}
          module={module}
          shapeId={index}
          piece={item}
          onUpdate={onUpdate}
        />
      ))}
    </div>
  );
};

export default PiecesSelector;

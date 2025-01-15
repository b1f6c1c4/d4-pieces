import React, {useCallback, useReducer, useState} from 'react';
import PiecePlacement from './PiecePlacement';

const PieceSelector = ({ module, shapeId, piece, onUpdate }) => {
  const [_, lolUpdate] = useReducer(a => a + 1, 0);

  const handleDecrease = useCallback(() => {
    if (piece.count > 0) {
      piece.count--;
      onUpdate(-1);
      lolUpdate();
    }
  }, [piece]);

  const handleIncrease = useCallback(() => {
    piece.count++;
    onUpdate(+1);
    lolUpdate();
  }, [piece]);

  return (
    <div className={`piece-selector ${piece.count ? '' : 'disabled'}`}>
      <div className="button-row">
        <button onClick={handleDecrease}>-</button>
        <input
          inputmode="numeric"
          value={piece.count}
          onChange={(e) => {
            const x = Number(e.target.value);
            const diff = x - piece.count;
            piece.count = x;
            onUpdate(diff);
            lolUpdate();
          }}
        />
        <button onClick={handleIncrease}>+</button>
      </div>
      <PiecePlacement
        {...{
          module,
          shapeId,
          piece,
          handleIncrease,
          handleDecrease,
        }}
      />
    </div>
  );
};

export default PieceSelector;

import React from 'react';
import Piece from './Piece';
import Board from './Board';

const Solution = ({ board, solution }) => {
  const { steps } = solution;

  return (
    <div className="solution">
      <Board shape={board} />
      <div className="overlay">
        {new Array(steps.size()).fill(0).map((_, id) => (
          <Piece
            key={id}
            stepId={id}
            shape={steps.get(id).shape}
            shapeId={steps.get(id).piece_id}
          />
        ))}
      </div>
    </div>
  );
};

export default Solution;

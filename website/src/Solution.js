import React from 'react';
import Piece from './Piece';
import Board from './Board';

const Solution = ({ pieces, board, solution }) => {
  const { steps } = solution;
  const overlays = [];
  new Array(steps.size()).fill(0).forEach((_, id) => {
    const s = steps.get(id);
    const sh = pieces[s.piece_id].shape;
    const st = sh['transform' + s.trs_id](false);
    const y = s.y - st.top;
    const x = s.x - st.left;
    overlays[s.piece_id] = (
      <Piece
      key={s.piece_id}
      stepId={id}
      shape={sh}
      shapeId={s.piece_id}
      transform={`matrix(${s.a}, ${s.b}, ${s.c}, ${s.d}, ${x * 30}, ${y * 30})`}
    />
    );
  });

  return (
    <div className="solution">
      <Board shape={board} />
      <div className="overlay">{overlays}</div>
    </div>
  );
};

export default Solution;

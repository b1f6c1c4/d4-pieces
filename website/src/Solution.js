import React from 'react';
import Piece from './Piece';
import Board from './Board';

const Solution = ({ pieces, board, solution }) => {
  const { steps } = solution;
  const used = [];
  const overlays = [];
  new Array(steps.size()).fill(0).forEach((_, id) => {
    const s = steps.get(id);
    if (overlays[s.piece_id]) // will be put in extra
      return;
    const pid = pieces[s.piece_id].i;
    const sh = pieces[s.piece_id].p.shape;
    const st = sh['transform' + s.trs_id](false);
    const y = s.y - st.top;
    const x = s.x - st.left;
    overlays[s.piece_id] = (
      <Piece
        key={s.piece_id}
        stepId={id}
        shape={sh}
        shapeId={pid}
        transform={`matrix(${s.a}, ${s.b}, ${s.c}, ${s.d}, ${x * 30}, ${y * 30})`}
      />
    );
  });
  let extraId = pieces.length;
  new Array(steps.size()).fill(0).forEach((_, id) => {
    const s = steps.get(id);
    if (!overlays[s.piece_id])
      return;
    const pid = pieces[s.piece_id].i;
    const sh = pieces[s.piece_id].p.shape;
    const st = sh['transform' + s.trs_id](false);
    const y = s.y - st.top;
    const x = s.x - st.left;
    overlays[extraId++] = (
      <Piece
        key={extraId}
        stepId={id}
        shape={sh}
        shapeId={pid}
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

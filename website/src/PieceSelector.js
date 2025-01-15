import React, {useReducer, useState} from 'react';
import Piece from './Piece';

const PieceSelector = ({ module, shapeId, piece }) => {
  const [override, setOverride] = useState(undefined);
  const [sym, setSym] = useState('D4');
  const [_, lolUpdate] = useReducer(a => a + 1, 0);

  const handleDecrease = () => {
    if (piece.count > 0) {
      piece.count--;
      lolUpdate();
    }
  };

  const handleIncrease = () => {
    piece.count++;
    lolUpdate();
  };

  const handleSym = (e) => {
    if (e.target.value === 'Custom') return;
    const grp = module.SymmetryGroup[e.target.value];
    for (let i = 0; i < 8; i++) {
      const placement = piece.placements.get(i);
      if (placement.duplicate) continue;
      placement.enabled = (grp.value & (1 << i));
      piece.placements.set(i, placement);
    }
    setSym(e.target.value);
  };
  const sg = piece.shape.classify.constructor.name.replace(/^SymmetryGroup_/, '');

  const transforms = [
    'matrix(+1, 0, 0,+1, 0, 0)',
    'matrix(-1, 0, 0,+1, 0, 0)',
    'matrix(+1, 0, 0,-1, 0, 0)',
    'matrix(-1, 0, 0,-1, 0, 0)',
    'matrix( 0,+1,+1, 0, 0, 0)',
    'matrix( 0,+1,-1, 0, 0, 0)',
    'matrix( 0,-1,+1, 0, 0, 0)',
    'matrix( 0,-1,-1, 0, 0, 0)',
  ];

  return (
    <div className={`piece-selector ${piece.count ? '' : 'disabled'}`}>
      <div className="button-row">
        <button onClick={handleDecrease}>-</button>
        <input
          inputmode="numeric"
          value={piece.count}
          onChange={(e) => { piece.count = Number(e.target.value); lolUpdate(); }}
        />
        <button onClick={handleIncrease}>+</button>
      </div>
      <Piece shape={piece.shape} reduced shapeId={shapeId} transform={override}
        onToggle={() => {
          if (piece.count < 2)
            piece.count = !piece.count;
          lolUpdate();
        }} />
      <div className="sym">
        {['Id', 'X', 'Y', '180', 'P', '90CW', '90CCW', 'S'].map((s, id) => {
          const placement = piece.placements.get(id);
          return (
            <input type="checkbox"
              key={id}
              title={s}
              disabled={placement.duplicate}
              checked={placement.enabled}
              onMouseEnter={() => setOverride(transforms[id])}
              onMouseLeave={() => setOverride()}
              onChange={(e) => {
                placement.enabled = e.target.checked;
                piece.placements.set(id, placement);
                setSym('Custom');
                lolUpdate();
              }}/>
          );
        })}
        <select onChange={handleSym} value={sym}>
          <option value="Custom">(Custom)</option>
          {['C1', 'C2', 'C4', 'D1_X', 'D1_Y', 'D1_P', 'D1_S', 'D2_XY', 'D2_PS', 'D4'].map(s => {
            const grp = module.SymmetryGroup[s];
            const prod = module.groupProduct(piece.shape.classify, grp)
            if (grp !== prod)
              return undefined;
            return (
              <option key={s} value={s}>{s}/{sg}</option>
            );
          })}
        </select>
      </div>
    </div>
  );
};

export default React.memo(PieceSelector);

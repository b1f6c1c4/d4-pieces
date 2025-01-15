import React, {useState} from 'react';
import Piece from './Piece';

const PiecePlacement = ({ module, shapeId, piece, handleIncrease, handleDecrease }) => {
  const [override, setOverride] = useState(undefined);
  const [sym, setSym] = useState('D4');

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
    <React.Fragment>
      <Piece shape={piece.shape} reduced shapeId={shapeId} transform={override}
        onToggle={() => {
          if (piece.count === 0)
            handleIncrease();
          else if (piece.count === 1)
            handleDecrease();
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
    </React.Fragment>
  );
};

export default React.memo(PiecePlacement);

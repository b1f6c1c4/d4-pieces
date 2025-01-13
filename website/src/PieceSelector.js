import React, {useReducer, useState} from 'react';
import Piece from './Piece';

const PieceSelector = ({ module, shapeId, piece, onChange }) => {
  const [override, setOverride] = useState(undefined);
  const [sym, setSym] = useState('D4');

  const handleDecrease = () => {
    if (piece.count > 0) {
      piece.count--;
      onChange();
    }
  };

  const handleIncrease = () => {
    piece.count++;
    onChange();
  };

  const handleSym = (e) => {
    if (e.target.value === 'Custom') return;
    const grp = module.SymmetryGroup[e.target.value];
    const prod = module.groupProduct(piece.shape.classify, grp)
    for (let i = 0; i < 8; i++) {
      const placement = piece.placements.get(i);
      if (placement.duplicate) continue;
      placement.enabled = (grp.value & (1 << i));
      piece.placements.set(i, placement);
    }
    setSym(e.target.value);
  };


  const containerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '10px'
  };

  const inputContainerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '5px'
  };

  return (
    <div style={containerStyle}>
      <Piece shape={override ?? piece.shape} reduced shapeId={shapeId} />
      <div style={inputContainerStyle}>
        <button onClick={handleDecrease}>-</button>
        <input
          type="number"
          value={piece.count}
          onChange={(e) => { piece.count = Number(e.target.value); onChange(); }}
          style={{ width: '50px', textAlign: 'center' }}
        />
        <button onClick={handleIncrease}>+</button>
      </div>
      <div className="sym">
        <span>Sym: {piece.shape.classify.constructor.name.replace(/^SymmetryGroup_/, '')}</span>
        {['Id', 'X', 'Y', '180', 'P', '90CW', '90CCW', 'S'].map((s, id) => {
          const placement = piece.placements.get(id);
          return (
            <input type="checkbox"
              key={id}
              title={s}
              disabled={placement.duplicate}
              checked={placement.enabled}
              onMouseEnter={() => setOverride(placement.normal)}
              onMouseLeave={() => setOverride()}
              onChange={(e) => {
                placement.enabled = e.target.checked;
                piece.placements.set(id, placement);
                setSym('Custom');
                onChange();
              }}/>
          );
        })}
        <span>Placement:</span>
        <select onChange={handleSym} value={sym}>
          <option value="Custom">(Custom)</option>
          {['C1', 'C2', 'C4', 'D1_X', 'D1_Y', 'D1_P', 'D1_S', 'D2_XY', 'D2_PS', 'D4'].map(s => {
            const grp = module.SymmetryGroup[s];
            const prod = module.groupProduct(piece.shape.classify, grp)
            if (grp == prod)
              return (
                <option key={s} value={s}>{s}</option>
              );
          })}
        </select>
      </div>
    </div>
  );
};

export default PieceSelector;

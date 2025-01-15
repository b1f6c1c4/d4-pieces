import './App.css';
import PiecesSelector from './PiecesSelector';
import Board from './Board';
import Solution from './Solution';
import {useEffect, useReducer, useState} from 'react';

const moduleLoader = window.Pieces({ locateFile: () => 'pieces.wasm' });

function App() {
  const [module, setModule] = useState();
  const [thinking, setThinking] = useState(true);
  const [knowns, setKnowns] = useState();
  const [pieces, setPieces] = useState([]);
  const [availCount, updateAvailCount] = useReducer(([p1, t1], b) =>
    b === undefined ? [0, 0] : [p1 + b[0], t1 + b[1]], [0, 0]);
  const [library, setLibrary] = useState([]);
  const [board, setBoard] = useState(undefined);
  const [solution, setSolution] = useState(undefined);
  const [solutionId, setSolutionId] = useState(0);

  useEffect(() => {
    moduleLoader.then((module) => {
      window.Pieces = module;
      setThinking(false);
      setModule(module);
      const knowns = {
        D: [ 0x20f, 0x307, 0x10f, 0x507, 0x30e, 0x10107, 0x10704, 0x707, ],
        T: [ 0x303, 0x207, 0x306, 0x107, 0xf, 0x307, 0x507, 0x10107, 0x707 ],
        J: [ 0x20f, 0x307, 0x10f, 0x507, 0x30e, 0x10701, 0x10107, 0x707 ],
        W: [ 0x306, 0x107, 0xf, 0x20f, 0x307, 0x10f, 0x507, 0x30e, 0x10701, 0x10704 ],
      };
      for (const key in knowns)
        knowns[key] = knowns[key]
          .map(v => new module.Shape(v).canonical_form(0b11111111).value);
      setKnowns(knowns);
      const pieces = [];
      let cnt = 0, tcnt = 0;
      for (let n = 1; n <= 8; n++) {
        const m = module.shape_count(n);
        for (let i = 0; i < m; i++) {
          const p = new module.Piece(module.shape_at(n, i));
          p.group = p.shape.classify;
          p.count = knowns.D.includes(p.shape.value);
          pieces.push(p);
          cnt += p.count;
          tcnt += p.count * p.shape.size;
        }
      }
      updateAvailCount(undefined);
      updateAvailCount([cnt, tcnt]);
      setPieces(pieces);
      setBoard(new module.Shape(window.BigInt('0b111011111110111111101111111011111110011111100111111')));
    });
  }, []);

  const handleToggle = (func, exact) => () => {
    let set;
    if (!exact) {
      const p0 = pieces.find(func);
      if (!p0) {
        console.error('Unexpected');
        return;
      }
      set = +!p0.count;
    }
    const next = [];
    let diff = 0, tdiff = 0;
    for (let i = 0; i < pieces.length; i++) {
      let p = pieces[i];
      const tgt = exact ? func(pieces[i]) : set;
      if ((exact || func(pieces[i])) && p.count !== tgt) {
        diff += tgt - p.count;
        tdiff += (tgt - p.count) * p.shape.size;
        p.count = tgt;
      }
      next.push(p);
    }
    setPieces(next);
    updateAvailCount([diff, tdiff]);
  };

  const handleSolve = (full) => () => {
    setSolution(undefined);
    setThinking(true);
    setTimeout(() => {
      const vec = new module.VPiece();
      const lib = [];
      pieces.forEach((p,i) => {
        if (p.count) {
          vec.push_back(p);
          lib.push({ p, i });
        }
      });
      setLibrary(lib);
      setSolution(module.solve(vec, board, full));
      setSolutionId(0);
      setThinking(false);
    }, 100);
  };

  return (
    <div className="App">
      <div>
        {board && (<Board shape={board} onToggle={(row, col) => {
          if (board.test(row, col))
            setBoard(board.clear(row, col));
          else
            setBoard(board.set(row, col));
        }} />)}
        {(solution && solution.size()) ? (
          <Solution pieces={library} board={board} solution={solution.get(solutionId)} />
        ) : undefined}
      </div>
      <div className="status-row">
        <span>{availCount[0]} pieces selected, {availCount[1]} / {board?.size} tiles!</span>
        <button disabled={thinking || !board || board.size > availCount[1]} onClick={handleSolve(true)}>Solve!</button>
        <button disabled={thinking || !board || board.size > availCount[1]} onClick={handleSolve(false)}>Solve All!</button>
        {solution && (
          <span>{solution.size()} Solutions found!</span>
        )}
        {(solution && solution.size() > 1) ? (
          <input
            type="number"
            max={solution.size() - 1}
            min={0}
            value={solutionId}
            onChange={(e) => setSolutionId(+e.target.value)}
            style={{ width: '50px', textAlign: 'center' }}
          />
        ) : undefined}
      </div>
      <div className="button-row">
        <button onClick={handleToggle((p) => knowns.D.includes(p.shape.value), true)}>D</button>
        <button onClick={handleToggle((p) => knowns.J.includes(p.shape.value), true)}>J</button>
        <button onClick={handleToggle((p) => knowns.T.includes(p.shape.value), true)}>T</button>
        <button onClick={handleToggle((p) => knowns.W.includes(p.shape.value), true)}>W</button>
        <div class="button">
          <span onClick={handleToggle((p) => true, true)}>All</span>
          <span onClick={handleToggle((p) => false, true)}>None</span>
          <button onClick={handleToggle((p) => p.shape.size === 1)}>1</button>
          <button onClick={handleToggle((p) => p.shape.size === 2)}>2</button>
          <button onClick={handleToggle((p) => p.shape.size === 3)}>3</button>
          <button onClick={handleToggle((p) => p.shape.size === 4)}>4</button>
          <button onClick={handleToggle((p) => p.shape.size === 5)}>5</button>
          <button onClick={handleToggle((p) => p.shape.size === 6)}>6</button>
          <button onClick={handleToggle((p) => p.shape.size === 7)}>7</button>
          <button onClick={handleToggle((p) => p.shape.size === 8)}>8</button>
        </div>
        <div class="button">
          <span onClick={handleToggle((p) => [module.SymmetryGroup.C1, module.SymmetryGroup.C2, module.SymmetryGroup.C4].includes(p.group) )}>C</span>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.C1   )}>1</button>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.C2   )}>2</button>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.C4   )}>4</button>
        </div>
        <div class="button">
          <span onClick={handleToggle((p) => [module.SymmetryGroup.D1_X, module.SymmetryGroup.D1_Y, module.SymmetryGroup.D1_P, module.SymmetryGroup.D1_S].includes(p.group) )}>D1</span>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.D1_X )}>X</button>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.D1_Y )}>Y</button>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.D1_P )}>P</button>
        </div>
        <div class="button">
          <span onClick={handleToggle((p) => [module.SymmetryGroup.D2_XY, module.SymmetryGroup.D2_PS].includes(p.group) )}>D2</span>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.D2_XY)}>XY</button>
          <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.D2_PS)}>PS</button>
        </div>
        <button onClick={handleToggle((p) => p.group === module.SymmetryGroup.D4   )}>D4</button>
      </div>
      <PiecesSelector
        module={module}
        pieces={pieces}
        onUpdate={updateAvailCount}
      />
    </div>
  );
}

export default App;

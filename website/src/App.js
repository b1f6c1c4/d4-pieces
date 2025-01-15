import './App.css';
import PiecesSelector from './PiecesSelector';
import Board from './Board';
import Solution from './Solution';
import {useEffect, useReducer, useState} from 'react';

const moduleLoader = window.Pieces({ locateFile: () => 'pieces.wasm' });

function App() {
  const [module, setModule] = useState();
  const [knowns, setKnowns] = useState();
  const [pieces, setPieces] = useState([]);
  const [availCount, updateAvailCount] = useReducer((a, b) => b === undefined ? 0 : a + b, 0);
  const [library, setLibrary] = useState([]);
  const [board, setBoard] = useState(undefined);
  const [solution, setSolution] = useState(undefined);
  const [solutionId, setSolutionId] = useState(0);

  useEffect(() => {
    moduleLoader.then((module) => {
      window.Pieces = module;
      setModule(module);
      const knowns = [
        0b1111000010000000,
        0b1110000000110000,
        0b011000000100000011000000,
        0b100000001000000011100000,
        0b1110000011100000,
        0b1110000011000000,
        0b1110000010100000,
        0b1111000000100000,
      ].map(v => new module.Shape(v).canonical_form(0b11111111).value);
      setKnowns(knowns);
      const pieces = [];
      let cnt = 0;
      for (let n = 1; n <= 8; n++) {
        const m = module.shape_count(n);
        for (let i = 0; i < m; i++) {
          const p = new module.Piece(module.shape_at(n, i));
          p.group = p.shape.classify;
          p.count = knowns.includes(p.shape.value);
          pieces.push(p);
          cnt += p.count;
        }
      }
      updateAvailCount(undefined);
      updateAvailCount(cnt);
      setPieces(pieces);
      setBoard(new module.Shape(window.BigInt('0b111011111110111111101111111011111110011111100111111')));
    });
  }, []);

  const handleToggle = (func) => () => {
    const p0 = pieces.find(func);
    if (!p0) {
      console.error('Unexpected');
      return;
    }
    const set = +!p0.count;
    const next = [];
    let diff = 0;
    for (let i = 0; i < pieces.length; i++) {
      let p = pieces[i];
      if (func(pieces[i]) && p.count !== set) {
        //p = p.clone();
        //p.group = pieces[i].group;
        diff += set - p.count;
        p.count = set;
      }
      next.push(p);
    }
    setPieces(next);
    updateAvailCount(diff);
  };

  const handleSolve = (full) => () => {
    setSolution(undefined);
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
        <span>{availCount} pieces selected!</span>
        <button onClick={handleSolve(true)}>Solve!</button>
        <button onClick={handleSolve(false)}>Solve All!</button>
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
        <button onClick={handleToggle((p) => knowns.includes(p.shape.value))}>Default</button>
        <div class="button">
          <span onClick={handleToggle((p) => true)}>All</span>
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

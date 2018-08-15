package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type (
	Board      uint64
	Row        uint16
	TransTable map[Board]TranTableEntry
)

type TranTableEntry struct {
	depth     uint16
	heuristic float64
}
type EvalState struct {
	transTable  TransTable
	maxDepth    int
	curDepth    int
	cacheHits   int
	movesEvaled uint32
	depthLimit  int
}

const (
	RowMask Board = 0xFFFF
	ColMask Board = 0x000F000F000F000F

	ScoreLostPenalty        float64 = 200000.0
	ScoreMonotonicityPower  float64 = 4.0
	ScoreMonotonicityWeight float64 = 47.0
	ScoreSumPower           float64 = 3.5
	ScoreSumWeight          float64 = 11.0
	ScoreMergesWeight       float64 = 700.0
	ScoreEmptyWeight        float64 = 270.0

	CprobThreshBase float64 = 0.0001
	CacheDepthLimit int     = 15
)

var (
	rowLeftTable  = make([]Row, 65536)
	rowRightTable = make([]Row, 65536)
	colUpTable    = make([]Board, 65536)
	colDownTable  = make([]Board, 65536)
	heuScoreTable = make([]float64, 65536)
	scoreTable    = make([]float64, 65536)
	r             = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func printBoard(board Board) {
	var i, j uint8
	for i = 0; i < 4; i++ {
		for j = 0; j < 4; j++ {
			var powerVal = uint8(board & 0xf)
			fmt.Printf("%6d", ternary(powerVal == 0, 0, 1<<powerVal))
			board >>= 4
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")
}

func unpackCol(row Row) Board {
	tmp := Board(row)
	return (tmp | (tmp << 12) | (tmp << 24) | (tmp << 36)) & ColMask
}

func reverseRow(row Row) Row {
	return (row >> 12) | ((row >> 4) & 0x00F0) | ((row << 4) & 0xF00) | (row << 12)
}

func transpose(x Board) Board {
	var a1 = x & 0xF0F00F0FF0F00F0F
	var a2 = x & 0x0000F0F00000F0F0
	var a3 = x & 0x0F0F00000F0F0000
	var a = a1 | (a2 << 12) | (a3 >> 12)
	var b1 = a & 0xFF00FF0000FF00FF
	var b2 = a & 0x00FF00FF00000000
	var b3 = a & 0x00000000FF00FF00
	return b1 | (b2 >> 24) | (b3 << 24)
}

func countEmpty(x Board) uint64 {
	x |= (x >> 2) & 0x3333333333333333
	x |= (x >> 1)
	x = ^x & 0x1111111111111111

	x += x >> 32
	x += x >> 16
	x += x >> 8
	x += x >> 4
	return uint64(x) & 0xf
}

func initTables() {
	var row uint16
	for row = 0; row < 65535; row++ {
		line := [4]uint16{
			(row >> 0) & 0xf,
			(row >> 4) & 0xf,
			(row >> 8) & 0xf,
			(row >> 12) & 0xf,
		}

		var score float64 = 0.0
		for _, rank := range line {
			if rank >= 2 {
				score += float64((rank - 1) * (1 << rank))
			}
		}
		scoreTable[row] = score

		var sum float64
		var empty int
		var merges int

		var prev int
		var counter int
		for _, rank := range line {
			sum += math.Pow(float64(rank), ScoreSumPower)
			if rank == 0 {
				empty++
			} else {
				if prev == int(rank) {
					counter++
				} else if counter > 0 {
					merges += 1 + counter
					counter = 0
				}
				prev = int(rank)
			}
		}
		if counter > 0 {
			merges += 1 + counter
		}

		var monotonicityLeft float64
		var monotonicityRight float64
		for i := 1; i < 4; i++ {
			if line[i-1] > line[i] {
				monotonicityLeft += math.Pow(float64(line[i-1]), ScoreMonotonicityPower) - math.Pow(float64(line[i]), ScoreMonotonicityPower)
			} else {
				monotonicityRight += math.Pow(float64(line[i]), ScoreMonotonicityPower) - math.Pow(float64(line[i-1]), ScoreMonotonicityPower)
			}
		}
		heuScoreTable[row] = ScoreLostPenalty +
			ScoreEmptyWeight*float64(empty) +
			ScoreEmptyWeight*float64(merges) -
			ScoreMonotonicityWeight*math.Min(monotonicityLeft, monotonicityRight) -
			ScoreSumWeight*sum

		for i := 0; i < 3; i++ {
			var j int
			for j = i + 1; j < 4; j++ {
				if line[j] != 0 {
					break
				}
			}
			if j == 4 {
				break
			}
			if line[i] == 0 {
				line[i] = line[j]
				line[j] = 0
				i--
			} else if line[i] == line[j] {
				if line[i] != 0xf {
					line[i]++
				}
				line[j] = 0
			}
		}
		var result = Row((line[0] << 0) | (line[1] << 4) | (line[2] << 8) | (line[3] << 12))
		revResult := reverseRow(result)
		revRow := reverseRow(Row(row))
		rowLeftTable[row] = Row(row) ^ result
		rowRightTable[revRow] = revRow ^ revResult
		colUpTable[row] = unpackCol(Row(row)) ^ unpackCol(result)
		colUpTable[revRow] = unpackCol(revRow) ^ unpackCol(revResult)
	}
}

func executeMove0(board Board) Board {
	ret := board
	t := transpose(board)
	ret ^= colUpTable[(t>>0)&RowMask] << 0
	ret ^= colUpTable[(t>>16)&RowMask] << 4
	ret ^= colUpTable[(t>>32)&RowMask] << 8
	ret ^= colUpTable[(t>>48)&RowMask] << 12
	return ret
}

func executeMove1(board Board) Board {
	ret := board
	t := transpose(board)
	ret ^= colDownTable[(t>>0)&RowMask] << 0
	ret ^= colDownTable[(t>>16)&RowMask] << 4
	ret ^= colDownTable[(t>>32)&RowMask] << 8
	ret ^= colDownTable[(t>>48)&RowMask] << 12
	return ret
}

func executeMove2(board Board) Board {
	ret := board
	ret ^= Board(rowLeftTable[(board>>0)&RowMask]) << 0
	ret ^= Board(rowLeftTable[(board>>16)&RowMask]) << 16
	ret ^= Board(rowLeftTable[(board>>32)&RowMask]) << 32
	ret ^= Board(rowLeftTable[(board>>48)&RowMask]) << 48
	return ret
}

func executeMove3(board Board) Board {
	ret := board
	ret ^= Board(rowRightTable[(board>>0)&RowMask]) << 0
	ret ^= Board(rowRightTable[(board>>16)&RowMask]) << 16
	ret ^= Board(rowRightTable[(board>>32)&RowMask]) << 32
	ret ^= Board(rowRightTable[(board>>48)&RowMask]) << 48
	return ret
}

var executeMove = [...]func(Board) Board{
	executeMove0,
	executeMove1,
	executeMove2,
	executeMove3,
}

func getMaxRank(board Board) int {
	var maxrank int
	for board != 0 {
		maxrank = maxInt(maxrank, int(board&0xf))
		board >>= 4
	}
	return maxrank
}

func countDistingTiles(board Board) int {
	var bitset uint16
	for board != 0 {
		bitset |= 1 << (board & 0xf)
		board >>= 4
	}

	bitset >>= 1
	var count int
	for bitset != 0 {
		bitset &= bitset - 1
		count++
	}
	return count
}

func scoreHelper(board Board, table []float64) float64 {
	return table[(board>>0)&RowMask] +
		table[(board>>16)&RowMask] +
		table[(board>>32)&RowMask] +
		table[(board>>48)&RowMask]
}

func scoreHeurBoard(board Board) float64 {
	return scoreHelper(board, heuScoreTable) + scoreHelper(transpose(board), heuScoreTable)
}

func scoreBoard(board Board) float64 {
	return scoreHelper(board, scoreTable)
}

func scoreMoveNode(state *EvalState, board Board, cprob float64) float64 {
	var best float64
	state.curDepth++
	for move := 0; move < 4; move++ {
		newBoard := executeMove[move](board)
		state.movesEvaled++

		if board != newBoard {
			best = math.Max(best, scoreTilechooseNode(state, newBoard, cprob))
		}
		state.curDepth--

	}
	return best
}

func scoreTilechooseNode(state *EvalState, board Board, cprob float64) float64 {
	if cprob < CprobThreshBase || state.curDepth >= state.depthLimit {
		state.maxDepth = maxInt(state.curDepth, state.maxDepth)
		return scoreHeurBoard(board)
	}
	if state.curDepth < CacheDepthLimit {
		entry, ok := state.transTable[board]
		if !ok {
			if entry.depth <= uint16(state.curDepth) {
				state.cacheHits++
				return entry.heuristic
			}
		}
	}

	numOpen := countEmpty(board)
	cprob /= float64(numOpen)

	var res float64
	tmp := board
	var tile2 Board = 1

	for tile2 != 0 {
		if (tmp & 0xf) == 0 {
			res += scoreMoveNode(state, board|tile2, cprob*0.9) * 0.9
			res += scoreMoveNode(state, board|(tile2<<1), cprob*0.1) * 0.1
		}
		tmp >>= 4
		tile2 <<= 4
	}
	res = res / float64(numOpen)

	if state.curDepth < CacheDepthLimit {
		entry := TranTableEntry{uint16(state.curDepth), res}
		state.transTable[board] = entry
	}
	return res
}

func _scoreToplevelMove(state *EvalState, board Board, move int) float64 {
	newboard := executeMove[move](board)
	if board == newboard {
		return 0
	}
	return scoreTilechooseNode(state, newboard, 1.0) + 1e-6
}

func scoreToplevelMove(board Board, move int) float64 {
	var state EvalState
	state.depthLimit = maxInt(3, countDistingTiles(board)-2)
	return _scoreToplevelMove(&state, board, move)
}

func findBestMove(board Board) int {
	var best float64
	var bestmove = -1

	for move := 0; move < 4; move++ {
		res := scoreToplevelMove(board, move)
		if res > best {
			best = res
			bestmove = move
		}
	}
	return bestmove
}

func drawTile() Board {
	return Board(ternary((r.Intn(10) < 9), 1, 2).(int))
}

func insertTileRand(board, tile Board) Board {
	var index int
	if countEmpty(board) != 0 {
		index = r.Intn(int(countEmpty(board)))
	}
	tmp := board
	for {
		for (tmp & 0xf) != 0 {
			tmp >>= 4
			tile <<= 4
		}
		if index == 0 {
			break
		}
		index--
		tmp >>= 4
		tile <<= 4
	}
	return board | tile
}

func initialBoard() Board {
	board := drawTile() << (4 * uint(r.Intn(16)))
	return insertTileRand(board, drawTile())
}

func playGame(getMove func(Board) int) {
	board := initialBoard()
	var moveno int
	var scorepenality int

	for {
		var newboard Board
		var move int

		for move = 0; move < 4; move++ {
			if executeMove[move](board) != board {
				break
			}
		}
		if move == 4 {
			break
		}

		move = getMove(board)
		if move < 0 {
			break
		}
		newboard = executeMove[move](board)
		if newboard == board {
			moveno--
			continue
		}

		tile := drawTile()
		if tile == 2 {
			scorepenality += 4
		}
		board = insertTileRand(newboard, tile)

	}
	printBoard(board)
}

func maxInt(a, b int) int {
	if b > a {
		return b
	}
	return a
}

func ternary(condition bool, a, b interface{}) interface{} {
	if condition {
		return a
	}
	return b
}

func main() {
	initTables()
	// fmt.Println(r.Intn(52))
	playGame(findBestMove)
}

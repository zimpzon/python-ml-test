﻿using System;
using System.Linq;

namespace Tixy
{
    public static class BoardConsolePrint3
    {
        public static void Print(IBoard board, string waitMsg = null, bool clear = true)
        {
            if (clear)
                Console.Clear();

            var allPieces = board.GetPlayerPieces();

            Console.Write("   ");
            for (int x = 0; x < Board.W; x++)
                Console.Write((char)('A' + x) + "  ");

            Console.WriteLine();
            for (int y = 0; y < Board.H; y++)
            {
                Console.Write((char)(y + '0' + 1));
                Console.Write("  ");
                for (int x = 0; x < Board.W; x++)
                {
                    var piece = allPieces.FirstOrDefault(p => p.X == x && p.Y == y);
                    Console.Write("|" + (piece == null ? " " : piece.Piece.Type.ToString()) + "|");
                }
                Console.WriteLine();
                Console.WriteLine("   " + new string('-', Board.W * 3));
            }
            if (!string.IsNullOrEmpty(waitMsg))
            {
                Console.WriteLine(waitMsg);
                Console.ReadKey();
            }
        }
    }
}

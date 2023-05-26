import random
import tkinter as tk
import threading
from .Enums import *


class GUIBoard:
    """
    基于Tkinter的GUI棋盘
    :Author: 韩昊轩
    :Date: 2023/05/13
    """

    def __init__(self, size, mode=GameType.PLAYER_VS_AI):
        self._mode = mode
        self._root = None
        self._canvas = None
        self._reset_button = None
        self._size = size
        self._loaded = False
        self._board = [[0 for _ in range(size)] for _ in range(size)]
        self._current_player = Player.BLACK
        self._thread = threading.Thread(target=self.__run)
        self._thread.start()
        self._reset_callback = None
        self._close_callback = None
        if mode == GameType.PLAYER_VS_AI:
            self._user_place: None | tuple[int, int, int] = None
            self._waiting_user_place = False
        self.step = 0

    def __del__(self):
        if self._root:
            self._root.quit()
            self._root = None
            if self._close_callback:
                self._close_callback()

    def __run(self):
        self._root = tk.Tk()
        self._root.title("五子棋")
        self._root.geometry(f"{self._size * 50 + 50}x{self._size * 50 + 120}")
        self._root.resizable(False, False)
        self._canvas = tk.Canvas(self._root, width=self._size * 50 + 50, height=self._size * 50 + 50, bg="#EBD5B5")
        self._canvas.pack()
        if self._mode == GameType.PLAYER_VS_AI or self._mode == GameType.PLAYER_VS_PLAYER:
            self._canvas.bind("<Button-1>", self.__click)
            self._reset_button = tk.Button(self._root, text="复位", command=self.reset, font=("微软雅黑", 13))
            # self._reset_button.pack()
        self._label3 = tk.Label(self._root, text="信息", font=("微软雅黑", 12))
        self._label3.place(x=5, y=self._size * 50 + 96)

        self.reset()
        self._root.protocol("WM_DELETE_WINDOW", self.__del__)
        self._label1 = tk.Label(self._root, text="韩昊轩 计科一班 201051040109", font=("微软雅黑", 12))
        self._label1.place(x=5, y=self._size * 50 + 52)
        self._label2 = tk.Label(self._root,
                                text="人机对弈模式" if self._mode == GameType.PLAYER_VS_AI else "自对弈学习模式",
                                font=("微软雅黑", 12))
        self._label2.place(x=5, y=self._size * 50 + 74)
        self._loaded = True
        self._root.mainloop()

    def __click(self, event):
        if self._mode == GameType.PLAYER_VS_AI and not self._waiting_user_place:
            return
        x, y = (event.x + 25) // 50 - 1, (event.y + 25) // 50 - 1
        if x >= self._size or y >= self._size or x < 0 or y < 0:
            return
        if self._board[y][x] == 0:
            self._board[y][x] = self._current_player
            # self.draw_piece(x, y, self._current_player)
            self._user_place = (x, y, self._current_player)
            # self._current_player = Player.SwitchPlayer(self._current_player)

    def __draw_board(self):
        for i in range(self._size):
            self._canvas.create_line(50, i * 50 + 50, self._size * 50, i * 50 + 50, width=2)
            self._canvas.create_line(i * 50 + 50, 50, i * 50 + 50, self._size * 50, width=2)
            self._canvas.create_text(i * 50 + 50, 10, text=str(i), font=("微软雅黑", 13))
            self._canvas.create_text(10, i * 50 + 50, text=str(i), font=("微软雅黑", 13))

    def draw_piece(self, x, y, player):
        """
        放置棋子
        :param x: x座标
        :param y: y座标
        :param player: 棋子玩家
        :return:
        """
        self.step += 1
        color = "#000000" if player == Player.BLACK else "#FFFFFF"
        self._canvas.create_oval(x * 50 + 30, y * 50 + 30, x * 50 + 70, y * 50 + 75, fill=color, outline=color)
        self._canvas.create_text(x * 50 + 50, y * 50 + 50, text=str(self.step), font=('微软雅黑', 15),
                                 fill='#FFFFFF' if player == Player.BLACK else '#000000')

    def reset(self):
        """
        重置棋盘
        :return:
        """
        self._board = [[0 for _ in range(self._size)] for _ in range(self._size)]
        self._canvas.delete("all")
        self._current_player = Player.BLACK
        self.__draw_board()
        self.step = 0
        if self._reset_callback:
            self._reset_callback()

    def wait(self):
        """
        等待GUI线程结束
        :return:
        """
        return self._thread.join()

    def title(self, title: str):
        """
        修改窗口标题
        :param title: 标题
        :return:
        """
        if self._root is not None:
            self._root.title(title)

    def wait_setup(self):
        """
        等待棋盘完成初始化
        :return:
        """
        while not self._loaded:
            pass

    def info(self, text: str):
        """
        在AI自对弈模式中为GUI添加提示信息
        :param text: 提示信息
        :return:
        """
        self._label3.config(text=text)

    def on_reset(self, cb):
        self._reset_callback = cb

    def on_close(self, cb):
        self._close_callback = cb

    def wait_place(self) -> tuple[int, int, int] | None:
        self._user_place = None
        self._waiting_user_place = True
        while self._user_place is None and self._root is not None:
            pass
        if self._root is None:
            return None
        assert self._user_place is not None
        self._waiting_user_place = False
        user_place = self._user_place
        self._user_place = None
        return user_place


if __name__ == "__main__":
    import time

    board = GUIBoard(15, mode=GameType.PLAYER_VS_PLAYER)
    board.wait_setup()
    print("GUI Board startup")
    board.wait()

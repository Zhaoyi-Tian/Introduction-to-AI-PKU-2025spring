import re
from Maze import Maze
from openai import OpenAI


# TODO: Replace this with your own prompt.
your_prompt = """
你接下面会进行一个吃豆人游戏，你可以移动吃豆人，你碰到墙或鬼后游戏就会结束，为简单起见，鬼不会移动，尽量在最短的步数内吃掉所有豆子，若不是必要的话尽量少走回头路，可使用A*搜索，以到所有豆子的距离为启发函数
"""

# Don't change this part.
output_format = """
输出必须是0-3的整数，上=0，下=1，左=2，右=3。
*重点*：(5,5)的上方是(4,5)，下方是(6,5)，左方是(5,4)，右方是(5,6)。
输出格式为：
“分析：XXXX。
动作：0（一个数字，不能出现其他数字）。”
"""

prompt = your_prompt + output_format


def get_game_state(maze: Maze, places: list, available: list) -> str:
    """
    Convert game state to natural language description.
    """
    description = ""
    for i in range(maze.height):
        for j in range(maze.width):
            description += f"({i},{j})="
            if maze.grid[i, j] == 0:
                description += f"空地"
            elif maze.grid[i, j] == 1:
                description += "墙壁"
            else:
                description += "豆子"
            description += ","
        description += "\n"
    places_str = ','.join(map(str, places))
    available_str = ','.join(map(str, available))
    state = f"""当前游戏状态（坐标均以0开始）：\n1、迷宫布局（0=空地,1=墙,2=豆子）：\n{description}\n2、吃豆人位置：{maze.pacman_pos[4]}\n3、鬼魂位置：{maze.pacman_pos[3]}\n4、曾经走过的位置：{places_str}\n5、可用方向：{available_str}\n"""
    return state


def get_ai_move(client: OpenAI, model_name: str, maze: Maze, file, places: list, available: list) -> int:
    """
    Get the move from the AI model.
    :param client: OpenAI client instance.
    :param model_name: Name of the AI model.
    :param maze: The maze object.
    :param file: The log file to write the output.
    :param places: The list of previous positions.
    :param available: The list of available directions.
    :return: The direction chosen by the AI.
    """
    state = get_game_state(maze, places, available)

    file.write("________________________________________________________\n")
    file.write(f"message:\n{state}\n")
    print("________________________________________________________")
    print(f"message:\n{state}")

    print("Waiting for AI response...")
    all_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": state
            }
        ],
        stream=False,
        temperature=.0
    )
    info = all_response.choices[0].message.content

    file.write(f"AI response:\n{info}\n")
    print(f"AI response:\n{info}")
    numbers = re.findall(r'\d+', info)
    choice = numbers[-1]
    return int(choice), info

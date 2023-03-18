import os
import sys
sys.path.append(os.path.dirname(__file__))
from opengait.opengait_main import main


def opengait_main():
    WORK_PATH = os.path.dirname(__file__)
    os.chdir(WORK_PATH)
    # print("WORK_PATH:", os.getcwd())
    res = main()
    return res


if __name__ == '__main__':
    opengait_main()
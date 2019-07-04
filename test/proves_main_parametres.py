#!/usr/bin/python

import sys

def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print arg
    print
    print str(sys.argv[1])

if __name__ == "__main__":
    main()

#per cridar-ho: python proves_main_parametres.py arg1 arg2 arg3
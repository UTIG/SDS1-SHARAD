#!/usr/bin/env python3
""" Test the prog progress class """
import prog

def main():
    # Instantiate as a test
    length1 = 35
    p = prog.Prog(length=length1, step=1, prog_symbol='*')
    for k in range(length1):
        p.print_Prog(k)
    p.close_Prog()

    kvals = 'abcdefghijklmnopqrstuvwxyz'
    p = prog.Prog(length=kvals, eta=False, etc=False, kind='%')
    p.set_prog_symbol('X')
    for k, x in enumerate(kvals):
        p.print_Prog(x)
    p.close_Prog()

if __name__ == "__main__":
    main()


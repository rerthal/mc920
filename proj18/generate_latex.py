# -*- coding: utf-8 -*-
import os

def print_preamble():
    print '\\documentclass{beamer}'
    print '\\usepackage{beamerthemeshadow}'
    print '\\usepackage[brazil]{babel}'
    print '\\usepackage[utf8]{inputenc}'
    print '\\usepackage{graphicx}'
    print '\\usepackage[compatibility=false]{caption}'
    print '\\usepackage{subcaption}'
    print
    print '\\def \\myheight {3cm}'
    print '\\def \\mywidth {3.5cm}'
    print
    print '\\begin{document}'
    print '\\title{MC920: Introdução ao Processamento de Imagem Digital}'
    print '\\author{Martin de Oliveira (118077) \\and Rafael Hermano (121286)}'
    print '\\date{\\today}'
    print
    print '\\frame{\\titlepage}'

def print_first(direc, f):
    prefix = f.split('.')[0]
    print '\\frame{\\frametitle{' + directory + '/' + f.replace('_', '\_') + '}'
    print '    \\begin{figure}[!ht]'
    print '        \\begin{subfigure}[ht]{0.2\\textwidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + directory + '/' + prefix + '.jpg}'
    print '            \\caption{Original}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + directory + '/' + prefix + '_directions.jpg}'
    print '            \\caption{Info. Direcional}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + directory + '/' + prefix + '_dir_small.jpg}'
    print '            \\caption{Janela 15x15}'
    print '        \\end{subfigure}'
    print '        \\\\'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + directory + '/' + prefix + '_dir_large.jpg}'
    print '            \\caption{Janela 45x45}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + directory + '/' + prefix + '_dir_rectified.jpg}'
    print '            \\caption{Retificado}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + directory + '/' + prefix + '_dir_rectified_pixels.jpg}'
    print '            \\caption{Pixels retificados}'
    print '        \\end{subfigure}'
    print '    \\end{figure}'
    print '}'

def print_second(direc, f):
    prefix = f.split('.')[0]
    print '\\frame{\\frametitle{' + directory + '/' + f.replace('_', '\_') + '}'
    print '    \\begin{figure}[!ht]'
    print '        \\begin{subfigure}[ht]{0.2\\textwidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '.jpg}'
    print '            \\caption{Original}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '_watershed.jpg}'
    print '            \\caption{Watershed}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '_water_dir.jpg}'
    print '            \\caption{Direção watershed}'
    print '        \\end{subfigure}'
    print '        \\\\'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '_dir_rectified.jpg}'
    print '            \\caption{Info. Direcional retificada (fig. original)}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{\\mywidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '_markers.jpg}'
    print '            \\caption{Marcadores}'
    print '        \\end{subfigure}'
    print '    \\end{figure}'
    print '}'

def print_final(direc, f):
    prefix = f.split('.')[0]
    print '\\frame{\\frametitle{' + directory + '/' + f.replace('_', '\_') + '}'
    print '    \\begin{figure}[!ht]'
    print '        \\begin{subfigure}[ht]{0.4\\textwidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '.jpg}'
    print '            \\caption{Original}'
    print '        \\end{subfigure}'
    print '        \\begin{subfigure}[ht]{0.4\\textwidth}'
    print '            \\includegraphics[height=\\myheight]{fingerprints/' + direc + '/' + prefix + '_reconnected.jpg}'
    print '            \\caption{Resultado final}'
    print '        \\end{subfigure}'
    print '    \\end{figure}'
    print '}'


if __name__ == '__main__':
    print_preamble()
    for directory in os.listdir("./fingerprints/"):
        for fingerprint in os.listdir("./fingerprints/" + directory):
            if '.tif' not in fingerprint:
                continue
            print_first(directory, fingerprint)
            print_second(directory, fingerprint)
            print_final(directory, fingerprint)
    print '\\end{document}'

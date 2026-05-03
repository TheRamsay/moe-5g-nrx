all: poster.pdf

poster.pdf: poster.tex
	pdflatex -interaction=nonstopmode poster.tex
	pdflatex -interaction=nonstopmode poster.tex

clean:
	rm -f *.aux *.log *.out *.toc poster.pdf

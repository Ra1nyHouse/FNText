all: fntext fntext_bi
fntext: fntext.c
	gcc fntext.c -o fntext -lm -fopenmp -O3
fntext_bi: fntext_bi.c
	gcc fntext_bi.c -o fntext_bi -lm -fopenmp -O3

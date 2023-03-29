/**
 * sparrow.c v 2014
 * This code is based on the code used in the SAT 2010 paper.
 * The code is simplified and improved.
 * Original code was based on gNovelty+ code, which seems
 * to be strongly based on the walksat code by H. Kautz and B. Selmann
 * Author: Adrian Balint
 */
//TODO: further possible optimizations:
//2. Literal represenation as in ubcsat

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/times.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <getopt.h>
#include <assert.h>
#include <signal.h> //gcc
//#include <iostream> //g++
//#include <csignal> //g++
/*----DEFINEs----*/
//use xor implementation of critVar
//#define XOR

//using namespace std; //g++

#define MAXCLAUSELENGTH 10000 //maximum number of literals per clause
#define STOREBLOCK  20000

# undef LLONG_MAX
#define LLONG_MAX  9223372036854775807
#define BIGINT long long int
#define MAXSCORE 300
#define getScore(VAR) (score[VAR])
#define setScore(VAR,VAL) (score[VAR]=VAL)
#define adjScore(VAR,VAL) (score[VAR]+=VAL)

#define incScore(VAR,CLS) (score[VAR]+=clauseWeight[CLS])
#define decScore(VAR,CLS) (score[VAR]-=clauseWeight[CLS])
#define BASECW 1 //base clause weight, when score used 1
#define GETOCCPOS(L) (2*abs(L)-(L<0))

/*--------*/

int (*pickClause)() = NULL;

/*----Instance data (independent from assignment)----*/
/** The numbers of variables. */
int numVars;
/** The number of clauses. */
int numClauses;
/** The number of literals. */
int numLiterals;
/** The value of the variables. The numbering starts at 1 and the possible values are 0 or 1. */
char *atom;
char *initialization_assign;
/** The clauses of the formula represented as: clause[clause_number][literal_number].
 * The clause and literal numbering start both at 0.*/
int **clause;
/**min and max clause length*/
int maxClauseSize;
int minClauseSize;
/** The number of occurrence of each literal.*/
int *numOccurrence;
/** The clauses where each literal occurs. For literal i : occurrence[i+MAXATOMS][j] gives the clause =
 * the j'th occurrence of literal i.  */
int **occurrence;
int maxNumOccurences = 0; //maximum number of occurences for a literal
/** neighbourVar[i] contains the array with all variables that are in the same clause with variable i*/
int **neighbourVar;
/*--------*/

/**----Assignment dependent data----*/
/** The number of false clauses.*/
int numFalse;
/** Array containing all clauses that are false. Managed as a list.*/
int *falseClause;
/** whereFalse[i]=j tells that clause i is listed in falseClause at position j.  */
int *whereFalseOrCritVar;
/** The number of true literals in each clause. */
unsigned short *numTrueLit;
//#ifdef XOR
/** whereFalse[i]=xor(all true literals of clause i)*/
//int *whereFalse;
//#else
/** whereFalse[i]=j tells that for clause i the variable j is critically responsible for satisfying i.*/
//int *whereFalse;
//#endif

/**score[i] tells how the overall weigths of the clauses will change if i is flipped*/
int bestVar;

int *score;
int *varLastChange;

/*----Gradient variables----*/
int gradient;
/**number of decresing varaibles*/
int numDecVar;
/** decVar is a list of all variables that when flipped decrease the number of unsat clauses. */
int *decVar;
int *isCandVar;
/*--------*/

/*----Weighting variables----*/
int sp; //smoothing probability
int *clauseWeight; //the weight of  each clause
/*this saves the extra weight the variables gets in a weighted algorithm instantiation*/
int numWeight; //the number of clauses with weight>1
int *weightedClause; //a list with all clauses with weight>1
int *whereWeight; //whereWeight[i]=j tells that clause i is at position j in weightedClause
int updateW;
/*--------*/

/**----Statistics variables----*/
unsigned BIGINT statNumGWalks=0;
unsigned BIGINT statNumSmooth=0; //how often a smoothing has been done.
unsigned BIGINT statNumWeight=0; //how often a weighting has been done.
unsigned BIGINT statXOROps=0;
unsigned BIGINT statClSearch=0;
BIGINT statdec1=0,statdec2=0,statdec3=0;//0<-1 1<-2 2-<3
BIGINT statinc0=0,statinc1=0,statinc2=0;//0->1 1->2 2->3
/*--------*/

/*----Sparrow variables----*/
/** Look-up table for the exponential function within Sparrow. The values are computed in the initSparrowProbs method.*/
double *scorePow;

/** contains the probabilities of the variables from an unsatisfied clause*/
double *probs;
double c0, c1, inv_c3;
int c2, c3;
unsigned short randomC=1; //Whether to pick a random unsat clause or iterate with flip counter
int keep_assig=0;
/*--------*/

/*----Input file variables----*/
FILE *fp;
char *fileName;
char *initialization_file;
/*---------*/

/** Run time variables variables*/
BIGINT seed;
BIGINT maxTries = LLONG_MAX;
BIGINT maxFlips = LLONG_MAX;
BIGINT flip;
int tryn;
BIGINT totalFlips=0,allowedFlips;
float timeOut = FLT_MAX;
int run = 1;
int printSol = 0;
double tryTime;
double totalTime = 0.;
long ticks_per_second;
int bestNumFalse;
//Sparrow parameters flags - indicates if the parameters were set on the command line
int c1_spec = 0, c2_spec = 0, c3_spec = 0, sp_spec = 0,sps_spec=0;
int randomCspec=0;
int sparray[9]= {0,100,600,1000,900,600,800,600,0}; //original setting
//int sparray[1000]= {0,100,600,1000,900,600,800,700,1000}; //original setting
int num_sp=9;
/*---------*/


/**-----luby sequence------*/
//every call of luby will return the next luby sequence
int luby_base=262144; //2^18
int use_luby = 0;
int lu = 1;
int lv = 1;
int luby() {
	if (((lu & (-lu)) == lv)) {
		lu = lu + 1;
		lv = 1;
		//printf("\n");
	} else {
		lv = 2 * lv;
	}
	return lv;
}
/**-----luby sequence------*/

int cpu_lim = -1;
void setCpuLimit() {
	if (cpu_lim != -1) {
		struct rlimit rl;
		getrlimit(RLIMIT_CPU, &rl);
		if (((rl.rlim_max == RLIM_INFINITY) || ((rlim_t) cpu_lim < rl.rlim_max))) {
			rl.rlim_cur = cpu_lim;
			if (setrlimit(RLIMIT_CPU, &rl) == -1)
				printf("c WARNING! Could not set resource limit:CPU-time.\n");
		}
	}
}

void printFormulaProperties() {
	fprintf(stderr, "c %-20s:  %s\n", "Instance name", fileName);
	if (initialization_file != NULL) {
		fprintf(stderr, "c %-20s:  %s\n", "Initialization file name", initialization_file);
	}
	fprintf(stderr, "c %-20s:  %d\n", "Number of variables", numVars);
	fprintf(stderr, "c %-20s:  %d\n", "Number of literals", numLiterals);
	fprintf(stderr, "c %-20s:  %d\n", "Number of Clauses", numClauses);
	fprintf(stderr, "c %-20s:  %d\n", "MaxNumOccurences", maxNumOccurences);
	fprintf(stderr, "c %-20s:  %d\n", "MaxClauseSize", maxClauseSize);
	fprintf(stderr, "c %-20s:  %d\n", "MinClauseSize", minClauseSize);
	fprintf(stderr, "c %-20s:  %6.4f\n", "Ratio", (float) numClauses / (float) numVars);
}
void printHeader() {
	fprintf(stderr, "---------------Sparrow 2014 SAT Solver---------------\n");
}

void printClauseMigrationCounters(){
	BIGINT total=statinc2+statdec3;
	printf("\nc migrations of clauses from x-sat to y-sat (#xor ops = #numTruelit ops):\n");
	printf("c %-30s: %-9lli (%6.2f X flips)\n", "total migration", total,(double)total/(double)flip);
	printf("c %-5s: %-6.4f  ", "0->1", (double) statinc0 / (double) total);
	printf(" %-5s: %-6.4f  ", "1->2", (double) statinc1 / (double) total);
	printf(" %-5s: %-6.4f\n", "2->3+", (double) (statinc2-statinc1-statinc0) / (double) total);
	printf("c %-5s: %-6.4f  ", "0<-1", (double) statdec1 / (double) total);
	printf(" %-5s: %-6.4f  ", "1<-2", (double) statdec2 / (double) total);
	printf(" %-5s: %-6.4f\n", "2<-3+", (double) (statdec3-statdec1-statdec2) / (double) total);
}
void printSolverParameters() {
	fprintf(stderr, "\nc Sparrow 2014 Parameteres: \n");
	fprintf(stderr, "c %-20s: %6.4f\n", "c1", c1);
	fprintf(stderr, "c %-20s: %d\n", "c2", c2);
	fprintf(stderr, "c %-20s: %d\n", "c3", c3);
	fprintf(stderr, "c %-20s: %5.3f\n", "smoothing prob.", ((float) sp / 1000));
	fprintf(stderr, "c %-20s: %lli\n", "seed", seed);
#ifdef XOR
	fprintf(stderr, "c %-20s: %-3s\n", "XOR implementation", "yes");
#else
	fprintf(stderr, "c %-20s: %-3s\n", "XOR implementation", "no");
#endif
	if (randomC)
		fprintf(stderr, "c %-20s: %-20s\n", "using:", "random clause selection");
	else
		fprintf(stderr, "c %-20s: %-20s\n", "using:", "flipCounter clause selection");
	fprintf(stderr, "c %-20s: %lli\n", "maxTries", maxTries);
	fprintf(stderr, "c %-20s: %lli\n", "maxFlips", maxFlips);
	fprintf(stderr, "c %-20s: %isec\n", "timeout", cpu_lim);
	if (use_luby){
		fprintf(stderr, "c %-20s: %-3s\n", "luby restarts", "yes");
		fprintf(stderr, "c %-20s: %d\n", "luby base", luby_base);
		fprintf(stderr, "c %-20s: %-3s\n", "keep assignement",keep_assig?"yes":"no" );
		int j=0;
		fprintf(stderr, "c %-20s: ", "sp sequence:" );
		for (;j<num_sp;j++){
			fprintf(stderr, "%4.3f, ",((float)sparray[j])/1000);
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "c %-20s: \n\n", "-->Starting solver");
	fflush(stderr);
}

void printSolution() {
	register int i;
	printf("v ");
	for (i = 1; i <= numVars; i++) {
		if (i % 21 == 0)
			printf("\nv ");
		if (atom[i] == 1)
			printf("%d ", i);
		else
			printf("%d ", -i);
	}
	printf("0\n");
}
//print the solutin on 1 line
void printSolution1line() {
	register int i;
	printf("v ");
	for (i = 1; i <= numVars; i++) {
		if (atom[i] == 1)
			printf("%d ", i);
		else
			printf("%d ", -i);
	}
	printf("0\n");
}

void printEndStatistics() {
	fprintf(stderr, "c EndStatistics:\n");

	fprintf(stderr, "c %-30s: %-9lli\n", "numFlips", flip);
	fprintf(stderr, "c %-30s: %-9lli (%4.2f%%) \n", "numProbWalks ", flip - statNumGWalks, 100 * (flip - statNumGWalks) / (float) flip);
	fprintf(stderr, "c %-30s: %-9lli (%4.2f%%) \n", "gradientWalks", statNumGWalks, 100 * (double) statNumGWalks / (double) flip);
	fprintf(stderr, "c %-30s: %-9lli (%4.2f%%) \n", "numSmooth ", statNumSmooth, 100 * (statNumSmooth) / (float) flip);
	fprintf(stderr, "c %-30s: %-9lli (%4.2f%%) \n", "numWeight ", statNumWeight, 100 * (statNumWeight) / (float) flip);
#ifndef XOR
	fprintf(stderr, "c %-30s: %-9lli (%8.2fX) \n", "statClSearch ", statClSearch, statClSearch/ (float) flip);
#endif

	fprintf(stderr, "c %-30s: %-8.3f\n", "Try CPU Time", tryTime);

	fprintf(stderr, "c %-30s: %-9i\n", "totalTries", tryn);
	fprintf(stderr, "c %-30s: %-9lli\n", "totalFlips", totalFlips);
	fprintf(stderr, "c %-30s: %-8.0f\n", "flips/sec", (double) totalFlips / totalTime );
	fprintf(stderr, "c %-30s: %-8.3fsec\n", "Total CPU Time", totalTime);
	//printClauseMigrationCounters();
	fflush(stderr);

}

void printStatsEndFlip() {
	if (numFalse < bestNumFalse) {
		//fprintf(stderr, "%8lli numFalse: %5d\n", flip, numFalse);
		bestNumFalse = numFalse;
	}
}



void allocateMemory() {
	// Allocating memory for the instance data (independent from the assignment).
	numLiterals = numVars * 2;
	atom = (char*) malloc(sizeof(char) * (numVars + 1));
	initialization_assign = (char*) malloc(sizeof(char) * (numVars + 1));
	clause = (int**) malloc(sizeof(int*) * (numClauses + 1));
	numOccurrence = (int*) malloc(sizeof(int) * (numLiterals + 1));
	occurrence = (int**) malloc(sizeof(int*) * (numLiterals + 1));
	neighbourVar = (int**) malloc(sizeof(int*) * numVars);

	// Allocating memory for the assignment dependent data.
	falseClause = (int*) malloc(sizeof(int) * (numClauses + 1));
	whereFalseOrCritVar = (int*) malloc(sizeof(int) * (numClauses + 1));
	numTrueLit = (unsigned short*) malloc(sizeof(unsigned short) * (numClauses + 1));
	score = (int*) malloc(sizeof(int) * (numVars + 1));

	decVar = (int*) malloc(sizeof(int) * (numVars + 1));
	isCandVar = (int*) malloc(sizeof(int) * (numVars + 1));

	varLastChange = (int*) malloc(sizeof(int) * (numVars + 1));

	clauseWeight = (int*) malloc(sizeof(int) * (numClauses + 1));
	weightedClause = (int*) malloc(sizeof(int) * (numClauses + 1));
	whereWeight = (int*) malloc(sizeof(int) * (numClauses + 1));
}

void parseFile() {
	register int i, j;
	int lit, r, clauseSize;
	int tatom;
	char c;
	int totalOcc=0;
	long filePos;
	int numNeighbours, cla, var;
	int *clptr;
	fp = NULL;
	fp = fopen(fileName, "r");
	if (fp == NULL) {
		fprintf(stderr, "c Error: Not able to open the file: %s", fileName);
		exit(-1);
	}

	// Start scanning the header and set numVars and numClauses
	for (;;) {
		c = fgetc(fp);
		if (c == 'c') //comment line - skip content
			do {
				c = fgetc(fp); //read the complete comment line until a eol is detected.
			} while ((c != '\n') && (c != EOF));
		else if (c == 'p') { //p-line detected
			if ((fscanf(fp, "%*s %d %d", &numVars, &numClauses))) //%*s should match with "cnf"
				break;
			break;
		} else {
			fprintf(stderr, "c No parameter line found! Computing number of atoms and number of clauses from file!\n");
			r = fseek(fp, -1L, SEEK_CUR); //try to unget c
			if (r == -1) {
				fprintf(stderr, "c Error: Not able to seek in file: %s", fileName);
				exit(-1);
			}
			filePos = ftell(fp);
			if (r == -1) {
				fprintf(stderr, "c Error: Not able to obtain position in file: %s", fileName);
				exit(-1);
			}

			numVars = 0;
			numClauses = 0;
			for (; fscanf(fp, "%i", &lit) == 1;) {
				if (lit == 0)
					numClauses++;
				else {
					tatom = abs(lit);
					if (tatom > numVars)
						numVars = tatom;
				}
			}
			fprintf(stderr, "c scanned numVars: %d numClauses: %d\n", numVars, numClauses);

			r = fseek(fp, filePos, SEEK_SET); //try to rewind the file to the beginning of the formula
			if (r == -1) {
				fprintf(stderr, "c Error: Not able to seek in file: %s", fileName);
				exit(-1);
			}

			break;
		}
	}
	// Finished scanning header.
	//allocating memory to use!
	allocateMemory();
	maxClauseSize = 0;
	minClauseSize = MAXCLAUSELENGTH;
	int *numOccurrenceT = (int*) malloc(sizeof(int) * (numLiterals + 1));

	int freeStore = 0;
	int *tempClause = 0;

	for (i = 0; i < numLiterals + 1; i++) {
		numOccurrence[i] = 0;
		numOccurrenceT[i] = 0;
	}

	for (i = 1; i <= numClauses; i++) {
			if (freeStore < MAXCLAUSELENGTH) {
				tempClause = (int*) malloc(sizeof(int) * STOREBLOCK);
				freeStore = STOREBLOCK;
			}
			clause[i] = tempClause;
			clauseSize = 0;
			do {
				r = fscanf(fp, "%i", &lit);
				if (lit != 0) {
					clauseSize++;
					*tempClause++ = lit;
					numOccurrenceT[GETOCCPOS(lit)]++;
					if (numOccurrenceT[GETOCCPOS(lit)] > maxNumOccurences)
						maxNumOccurences = numOccurrenceT[GETOCCPOS(lit)];
					totalOcc++;
				} else {
					*tempClause++ = 0; //0 sentinel as literal!
				}
				freeStore--;
			} while (lit != 0);
			if (clauseSize > maxClauseSize)
				maxClauseSize = clauseSize;
			if (clauseSize < minClauseSize)
				minClauseSize = clauseSize;
		}


	occurrence[0] = (int*) malloc(sizeof(int) * (totalOcc + numLiterals+2));
		int occpos=0;
		for (i = 0; i < numLiterals + 1; i++) {
			occurrence[i] = (occurrence[0]+occpos);
			occpos+=numOccurrenceT[i] + 1;
		}

		for (i = 1; i <= numClauses; i++) {
			j = 0;
			while ((lit = clause[i][j])) {
				occurrence[GETOCCPOS(lit)][numOccurrence[GETOCCPOS(lit)]++] = i;
				j++;
			}
		}
		//end occurrence array with a sentinel
		for (lit=1;lit<=numVars;lit++){
			occurrence[GETOCCPOS(lit)][numOccurrence[GETOCCPOS(lit)]] = 0;
			occurrence[GETOCCPOS(-lit)][numOccurrence[GETOCCPOS(-lit)]] = 0;
		}

	//Now the maximum size of a clause is determined!
	probs = (double*) malloc(sizeof(double) * (maxClauseSize + 1));

	//Constructing the neighbor array from the occurrence-arrays .
	freeStore = 0;
	int *tempNeighbour = 0;
	int isNeighbour[numVars + 1]; //isNeighbour[j]=i means that j is a neighbor-var of i.
	int *occptr;
	for(i = 1; i <= numVars; i++)
	isNeighbour[i] = 0;
	for (i = 1; i <= numVars; i++) {
		numNeighbours = 0;
		//first take a look at all positive occurrences of i
		//for (j = 0; j < numOccurrence[GETOCCPOS(i)]; j++) {
		occptr = &occurrence[GETOCCPOS(i)][0];
		while ((cla = *occptr)){
			//cla = occurrence[GETOCCPOS(i)][j];
			clptr = &clause[cla][0];
			while ((var = abs(*clptr))) {
				if ((isNeighbour[var] != i) && (var != i)) { //if it is not all ready marked as a neighbor of i mark it.
					isNeighbour[var] = i;
					numNeighbours++;
				}
				clptr++;
			}
			occptr++;
		}
		//then take a look at all negative occurrence of i
		//for (j = 0; j < numOccurrence[GETOCCPOS(-i)]; j++) {
		occptr = &occurrence[GETOCCPOS(-i)][0];
		while ((cla = *occptr)){
			//cla = occurrence[GETOCCPOS(-i)][j];
			clptr = &clause[cla][0];
			while ((var = abs(*clptr))) {
				if ((isNeighbour[var] != i) && (var != i)) { //if it is not all ready marked as a neighbor of i mark it.
					isNeighbour[var] = i;
					numNeighbours++;
				}
				clptr++;
			}
			occptr++;
		}
		if (freeStore < numNeighbours + 1) {
			tempNeighbour = (int*) malloc(sizeof(int) * STOREBLOCK);
			freeStore = STOREBLOCK;
		}
		neighbourVar[i] = tempNeighbour;
		freeStore -= numNeighbours + 1;
		if (numNeighbours >= 1) {
			for (j = 1; j <= numVars; j++) {
				if (isNeighbour[j] == i) {
					*(tempNeighbour++) = j;
					numNeighbours--;
					if (numNeighbours == 0) {
						*(tempNeighbour++) = 0;
						break;
					}
				}
			}
		} else
			*(tempNeighbour++) = 0;
	}
	free(numOccurrenceT);
	fclose(fp);

	if (initialization_file != NULL) {
		fp = NULL;
		fp = fopen(initialization_file, "r");
		if (fp == NULL) {
			fprintf(stderr, "c Error: Not able to open the file: %s", initialization_file);
			exit(-1);
		}
		int assign;
		for (i = 1; i <= numVars; i++) {
			fscanf(fp, "%d", &assign);
			initialization_assign[i] = assign;
		}
		fprintf(stderr, "c Hint: Initialization file load successfully!\n");
		fclose(fp);
	}
}

void init() {
	register int i;
	int critLit = 0, lit;
	int *clptr;
	ticks_per_second = sysconf(_SC_CLK_TCK);
	statNumGWalks = 0; //how often a gradient walk was done
	statNumSmooth = 0; //how often a smoothing has been done.
	statNumWeight = 0; //how often a smoothing has been done.
	statXOROps = 0;
	statClSearch=0;
	numFalse = 0;
	numWeight = 0;

	for (i = 1; i <= numVars; i++) {
		if ((!keep_assig)){//new assigment should be generated
			if (initialization_file != NULL) {
				atom[i] = initialization_assign[i];
			}
			else{
				atom[i] = rand() % 2;
			}
		}
		score[i] = 0;
		varLastChange[i] = -1; //-1 means never changed
	}
	//pass trough all clauses and apply the assignment previously generated
	for (i = 1; i <= numClauses; i++) {
		clptr = & clause[i][0];
		numTrueLit[i] = 0;
		whereWeight[i] = -1;
		whereFalseOrCritVar[i] = 0;
		while ((lit = *clptr)) {
			if (atom[abs(lit)] == (lit > 0)) {
				numTrueLit[i]++;
				critLit = lit;
#ifdef XOR
				whereFalseOrCritVar[i] ^= abs(lit);
#endif
			}
			clptr++;
		}
		if (numTrueLit[i] == 1) {
			//if the clause has only one literal that causes it to be sat,
			//then this var. will break the sat of the clause if flipped.
#ifndef XOR
			whereFalseOrCritVar[i] = abs(critLit);
#endif
			score[abs(critLit)]--;
		} else if (numTrueLit[i] == 0) {
			//add this clause to the list of unsat caluses.
			falseClause[numFalse] = i;
			whereFalseOrCritVar[i] = numFalse;
			numFalse++;

			//if the clause is unsat fliping any variable from it will make it sat
			//-> increase the score of all variables within this clause
			clptr = & clause[i][0];
			while ((lit = *clptr)) {
				score[abs(lit)]++;
				clptr++;
			}
		}
		clauseWeight[i] = BASECW;
	}
	numDecVar = 0;
	//add all variables that are decreasing to the list of decreasing variables.
	for (i = 1; i <= numVars; i++)
		if (getScore(i) > 0) {
			decVar[numDecVar] = i;
			numDecVar++;
			isCandVar[i] = 1;
		} else {
			isCandVar[i] = 0;
		}
}

/** Checks whether the assignment from atom is a satisfying assignment.*/
int  checkAssignment() {
	register int i;
	int sat, lit;
	int *clptr;
	for (i = 1; i <= numClauses; i++) {
		sat = 0;
		clptr =&clause[i][0];
		while ((lit = (*clptr))) {
			if (atom[abs(lit)] == (lit > 0)){
				sat = 1;
				break;
			}
		clptr++;
		}
		if (sat == 0) {
			fprintf(stderr, "\nClause %d is unsatified by assignment\n", i);
			return 0;
		}
	}
	return 1;
}

void  smooth2() { //for all weighted !!!satisfied!!! clauses decrease the score by 1
	register int i, c, var;
	for (i = 0; i < numWeight; i++) {
		c = weightedClause[i];
		if (numTrueLit[c] > 0) {
			if (--clauseWeight[c] == BASECW) { //remove from the list of weighted clauses
				numWeight--;
				weightedClause[i] = weightedClause[numWeight];
				whereWeight[weightedClause[i]] = i;
				whereWeight[c] = -1;
				i--;
			}
			if (numTrueLit[c] == 1) {
				var = whereFalseOrCritVar[c];
				//clause lost one weight and whereFalse had this weight as negative, so we have to add one to the weigth of var
				adjScore(var, 1);
				if ((getScore(var) > 0) && (!isCandVar[var]) && (varLastChange[var] < flip - 1)) {
					isCandVar[var] = 1;
					decVar[numDecVar] = var;
					numDecVar++;
				}
			}
		}
	}
	statNumSmooth++;
}

void  updateWeights() { //for all unsat clauses increase the weight by 1.
	statNumWeight++;
	int i, j;
	int c, var;
	for (i = 0; i < numFalse; i++) {
		c = falseClause[i];
		clauseWeight[c]++;
		if ((whereWeight[c] == -1) && (clauseWeight[c] > BASECW)) { //add to the list of weigthedClause
			weightedClause[numWeight] = c;
			whereWeight[c] = numWeight;
			numWeight++;
		}
		j = 0;
		while ((var = abs(clause[c][j]))) {
			score[var]++;
			if ((!isCandVar[var]) && (getScore(var) > 0) && (varLastChange[var] < flip - 1)) {
				isCandVar[var] = 1;
				decVar[numDecVar] = var;
				numDecVar++;
			}
			j++;
		}
	}
}

//pick a clause with the flip counter and not randomly
static int  pickClauseRandom(){
	return rand() % numFalse;
}

static int pickClauseF(){
	return flip % numFalse;
}

void pickVar() {
	register int i, j;
	int var;
	int bestScore = -numClauses;
	double probAge = 1.0, baseAge;
	int varChanged = -1;
	int scoreVar;
	int *clptr;
	int rClause; //randomly choosen clause.
	//g2Wsat part - the greedy part - if there is a variable that decreases the number of variables then choose the best one.

	if (numDecVar > 0) {
		//find the variable with the best score, and the following variable with same score.
		for (i = 0; i < numDecVar; i++) {
			var = decVar[i];
			scoreVar = getScore(var);
			if (scoreVar > 0) {
				if (bestScore < scoreVar) {
					bestScore = scoreVar;
					bestVar = var;
					varChanged = varLastChange[var];
				} else if (bestScore == scoreVar) //found one with the same score
					if (varLastChange[var] < varChanged) { //check if it is younger
						bestVar = var; //this var being younger is chosen.
						varChanged = varLastChange[var];
					}
			} else {
				numDecVar--;
				decVar[i] = decVar[numDecVar];
				//whereDecVar[decVar[numDecVar]]=i;
				isCandVar[var] = 0;
				i--;
			}
		}
	}

	if (bestScore != -numClauses) {
		statNumGWalks++;
	} else {
		//new probability distribution replacing adaptNovelty+
		//a variable is YOUNG if it was flipped not long time ago
		//a variable is OLD if it was flipped long ago in the past, or not at all
		rClause = falseClause[pickClause()];
		double sumProb = 0;
		i = 0;
		clptr = &clause[rClause][0];
		while ((var = abs(*clptr))) {
			scoreVar = getScore(var);
			if (scoreVar < -300) //has to be limited because the score is weighted
				probs[i] = scorePow[300];
			else {
				if (scoreVar > 0) {
					probs[i] = 1.0;
				} else {
					probs[i] = scorePow[abs(scoreVar)];
				}
			}
			baseAge = (double) (flip - varLastChange[var]) * inv_c3;
			for (j = 0, probAge = 1.0; j < c2; j++)
				probAge *= baseAge;
			probAge += 1.0;
			probs[i] *= probAge;
			sumProb += probs[i];
			i++;
			clptr++;
		}
		double randPosition = (double) (rand()) / (RAND_MAX+1.0) * sumProb;
		for (i = i-1; i!=0; i--) {
			sumProb -= probs[i];
			if (sumProb <= randPosition)
				break;
		}


		bestVar = abs(clause[rClause][i]);

		if (sp < 1000) {
			if (rand() % 1000 < sp)
				smooth2();
		else
			updateWeights();
		}
	}
	return;
}

void flipAtom() {
	int var;
	int *ocptr; //occurrence pointer
	int *clptr; //clause pointer
	int tClause; //temporary clause variable
	int xMakesSat; //tells which literal of x will make the clauses where it appears sat.
	if (atom[bestVar] == 1)
		xMakesSat = -bestVar; //if x=1 then all clauses containing -x will be made sat after fliping bestVar
	else
		xMakesSat = bestVar; //if x=0 then all clauses containing x will be made sat after fliping bestVar

	atom[bestVar] = 1 - atom[bestVar];
	//all Neighbours of x with score>0 are considered candVars without taking into account if they are in decVar or not.
	//trough this mechanism we can avoid that a variable that was fliped and increased the number of false variable is added to the
	//decVar array - this variable is not promissing.

	clptr=&neighbourVar[bestVar][0];
	while ((var = abs(*clptr))) {
		isCandVar[var] = (getScore(var) > 0);
		clptr++;
	}

	//1. all clauses that contain the literal xMakesSat will become SAT, if they where not already sat.
	ocptr=&occurrence[GETOCCPOS(xMakesSat)][0];
	while ((tClause = *ocptr)) {
		//tClause = occurrence[xMakesSat + numVars][i];
		//if the clause is unsat it will become SAT so it has to be removed from the list of unsat-clauses.
		if (numTrueLit[tClause] == 0) {
			//remove from unsat-list
			falseClause[whereFalseOrCritVar[tClause]] = falseClause[--numFalse]; //overwrite this clause with the last clause in the list.
			whereFalseOrCritVar[falseClause[numFalse]] = whereFalseOrCritVar[tClause];
			whereFalseOrCritVar[tClause] = 0;
#ifndef XOR
			whereFalseOrCritVar[tClause] = bestVar; //this variable is now critically responsible for satisfying tClause
#endif
			//adapt the scores of the variables
			//the score of x has to be decreased by one because x is critical and will break this clause if fliped.
			decScore(bestVar, tClause);
			//the scores of all variables from tClause have to be decreased by one because tClause is not UNSAT any more
			//j = 0;
			statinc0++;
			clptr= &clause[tClause][0];
			while ((var = abs(*clptr))) {
				decScore(var, tClause);
				//j++;
				clptr++;
			}
		} else {
			//if the clause is satisfied by only one literal then the score has to be increased by one for this var.
			//because fliping this variable will no longer break the clause
			if (numTrueLit[tClause] == 1) {
				incScore(whereFalseOrCritVar[tClause], tClause);
				statinc1++;
			}
		}
		//if the number of numTrueLit[tClause]>=2 then nothing will change in the scores
		numTrueLit[tClause]++; //the number of true Lit is increased.
		statinc2++;
#ifdef XOR
		whereFalseOrCritVar[tClause] ^= bestVar;
#endif
		ocptr++;
	}

	//2. all clauses that contain the literal -xMakesSat=0 will not be longer satisfied by variable x.
	//all this clauses contained x as a satisfying literal
	//i = 0;
	ocptr=&occurrence[GETOCCPOS(-xMakesSat)][0];
	while ((tClause = *ocptr)) {
#ifdef XOR
		whereFalseOrCritVar[tClause] ^= bestVar;
#endif
		if (numTrueLit[tClause] == 1) { //then xMakesSat=1 was the satisfying literal.
			//this clause gets unsat.
			falseClause[numFalse] = tClause;
			whereFalseOrCritVar[tClause] = numFalse;
			numFalse++;
			//the score of x has to be increased by one because it is not breaking any more for this clause.
			incScore(bestVar, tClause);
			statdec1++;
			//the scores of all variables have to be increased by one ; inclusive x because flipping them will make the clause again sat

			clptr = &clause[tClause][0];
			while ((var = abs(*clptr))) {
				incScore(var, tClause);
				clptr++;
			}
		} else if (numTrueLit[tClause] == 2) { //find which literal is true and make it critical and decrease its score
			statdec2++;
#ifdef XOR
			decScore(whereFalseOrCritVar[tClause],tClause);
#else
			clptr = &clause[tClause][0];
			while ((var = abs(*clptr))) {
				statClSearch++;
				if (((*clptr > 0) == atom[var])) { //x can not be the var anymore because it was flipped //&&(xMakesSat!=var)
					whereFalseOrCritVar[tClause] = var;
					decScore(var, tClause);
					break;
				}
				clptr++;
			}
#endif
		}
		numTrueLit[tClause]--;
		statdec3++;
		ocptr++;
	}

	//acoordant to G2WSAT only the scores of variables within neighbourVar[x] have changed.
	clptr= &neighbourVar[bestVar][0];
	while ((var = *clptr)) {
		if ((getScore(var) > 0) && (!isCandVar[var])) { //is not in the list of decreasing variables
			//add to decVar
			decVar[numDecVar] = var;
			//whereDecVar[var]=numDecVar;
			numDecVar++;
		}
		clptr++;
	}
}

double elapsed_seconds(void) {
	double answer;
	static struct tms prog_tms;
	static long prev_times = 0;
	(void) times(&prog_tms);
	answer = ((double) (((long) prog_tms.tms_utime) - prev_times)) / ((double) ticks_per_second);
	prev_times = (long) prog_tms.tms_utime;
	return answer;
}

void printUsage() {//TODO: stimmt nicht mehr so ganz, überarbeiten
	printf("\nSparrow version 2014\n");
	printf("Code Authors: Adrian Balint\n");
	printf("Algo Authors: Adrian Balint & Andreas Fröhlich\n");
	printf("Citation: Adrian Balint & Andreas Fröhlich: Improving Stochastic Local Search for SAT with a New Probability Distribution, SAT 2010\n");
	printf("Ulm University - Institute of Theoretical Computer Science\n");
	printf("----------------------------------------------------------\n");
	printf("\nUsage of sparrow:\n");
	printf("./sparrow [options] <DIMACS CNF instance> [<seed>]\n");
	printf("\nSparrow options:\n");
	printf("-f or --initialization <file> : variable initialization (default: random)\n");
	printf("--c1 <double_value> : c1 constant from the Sparrow heuristic (default: 3sat:2.15; 5sat:2.855; 7sat:6.5)\n");
	printf("--c2 <int_value> : c2 constant from the Sparrow heuristic (default: 3sat,5sat,7sat:4)\n");
	printf("--c3 <int_value> : c3 constant from the Sparrow heuristic (default: 3sat,7sat:10⁵; 5sat:0.75*10⁵)\n");
	printf("--sp <double_value> : smoothing probability inherited from gNovelty+ heuristic (default: 3sat:0.347; 5sat:1.0; 7sat:0.83)"
			"(values between [0..1.0] only the first 3 digits after dot will be taken into account)\n");
	printf("\nFurther options:\n");
	printf("--maxflips <int_value> : maximum number of flips (default: LLONG_MAX)\n");
	printf("--runs <int_value> : number of tries to solve the problem (default: LLONG_MAX)\n");
	printf("--luby : use luby interval restarts (default: off)\n");
	printf("--luby_base <int_value> : number of flips per base unit (default: 2^18)\n");
	printf("--sps <float_value> : of sp values for sequence (default: 0.,.1,.6,1.,.9,.6,.8,.7,1.)\n");
	printf("--randomc  : random clause selection (default: on)\n");
	printf("-a or --printsolution : print solution (default: off)\n");
	printf("-h or --help : print usage \n\n");
}

void initSparrow() {
	scorePow = (double*) malloc(sizeof(double) * (MAXSCORE + 1));
	int i;
	for (i = 0; i <= MAXSCORE; i++) {
		scorePow[i] = pow(c1, -i);
	}
	inv_c3 = 1. / c3;
	
	if (initialization_file != NULL) {
		atom[i] = initialization_assign[i];
	}
}

void parseParameters(int argc, char *argv[]) {
	double spParam;
	//define the argument parser
	static const struct option long_options[] = {
		{ "initialization", required_argument, 0, 'f' },
		{ "luby_base", required_argument, 0, 'u' }, 
		{ "luby", no_argument, 0, 'l' }, 
		{ "timeout", required_argument, 0, 11 }, 
		{ "randomc", required_argument, 0, 'r' }, 
		{ "c1", required_argument, 0, 'b' }, 
		{ "c2", required_argument, 0, 'e' }, 
		{ "c3", required_argument, 0, 'd' }, 
		{ "sp", required_argument, 0, 'p' }, 
		{ "runs", required_argument, 0, 't' }, 
		{ "maxflips", required_argument, 0, 'm' },
		{ "printSolution", no_argument, 0, 'a' },
		{ "help", no_argument, 0, 'h' }, 
		{ 0, 0, 0, 0 } 
	};

	while (optind < argc) {
		int index = -1;
		struct option * opt = 0;
		int result = getopt_long(argc, argv, "b:e:d:p:t:m:ahr:lu:kf:", long_options, &index); //
		if (result == -1)
			break; /* end of list */
		switch (result) {
		case 'f':
			initialization_file = optarg;
			break;
		case 'u': // luby_base
			luby_base = atoi(optarg);
			break;
		case 11: // timelimit in seconds
			cpu_lim = atoi(optarg);
			break;
		case 'k': //keep assignment after restart
			keep_assig = 1;
			break;
		case 'l': //use luby restarts?
			use_luby = 1;
			break;
		case 'h':
			printUsage();
			exit(0);
			break;
		case 'b': //this stands for c1
			c1 = atof(optarg);
			c1_spec = 1;
			break;
		case 'e': //this stands for c2
			c2 = atoi(optarg);
			c2_spec = 1;
			break;
		case 'd': //this stands for c3
			c3 = atoi(optarg);
			c3_spec = 1;
			break;
		case 'p': //this stands for sp
			spParam = atof(optarg);
			sp_spec = 1;
			sp = spParam * 1000;
			break;
		case 'r': //select clause randomly
			randomC = atoi(optarg);
			randomCspec = 1;
			break;
		case 't': //maximum number of tries to solve the problems within the cutoff
			maxTries = strtol(optarg, NULL, 10);
			break;
		case 'm': //maximum number of flips to solve the problem
			maxFlips = strtol(optarg, NULL, 10);
			break;
		case 'a': //print assigment of variables
			printSol = 1;
			break;
		case 0: /* all parameter that do not */
			/* appear in the optstring */
			opt = (struct option *) &(long_options[index]);
			printf("'%s' was specified.", opt->name);
			if (opt->has_arg == required_argument)
				printf("Arg: <%s>", optarg);
			printf("\n");
			break;
		default:
			printf("parameter not known!\n");
			printUsage();
			exit(0);
			break;
		}
	}
	if (optind == argc) {
		printf("ERROR: You have to specify at least an instance file!\n");
		printUsage();
		exit(0);
	}
	fileName = *(argv + optind);

	if (argc > optind + 1) {
		seed = atoi(*(argv + optind + 1));
		if (seed == 0)
			printf("c there might be an error in the command line or is your seed 0?\n");
	} else
		seed = time(0);
}

//void handle_interrupt(int signum) { //if using g++

void handle_interrupt() { //if using gcc
	fprintf(stderr, "\nc Cought signal... Exiting\n ");
	tryTime = elapsed_seconds();
	fprintf(stderr, "\nc UNKNOWN best(%d) (%-15.5fsec)\n", bestNumFalse, tryTime);
	printEndStatistics();
	fflush(NULL);
	exit(-1);
}

void setupSignalHandler() {
	signal(SIGTERM, handle_interrupt);
	signal(SIGINT, handle_interrupt);
	signal(SIGQUIT, handle_interrupt);
	signal(SIGABRT, handle_interrupt);
	signal(SIGKILL, handle_interrupt);
	signal(SIGXCPU, handle_interrupt);

}

void setupSparrowParameters() {
	if (maxClauseSize <= 3) {
		fprintf(stderr, "c %-20s:  %s\n", "Parameter setting ", "3 SAT");
		if (!c1_spec)
			c1 = 2.15;
		if (!c2_spec)
			c2 = 4;
		if (!c3_spec)
			c3 = 100000;
		if (!sp_spec)
			sp = 347;
	} else if (maxClauseSize <= 5) {
		fprintf(stderr, "c %-20s:  %s\n", "Parameter setting ", "5 SAT");
		if (!c1_spec)
			c1 = 2.85;
		if (!c2_spec)
			c2 = 4;
		if (!c3_spec)
			c3 = 75000;
		if (!sp_spec)
			sp = 1000;
	} else {
		fprintf(stderr, "c %-20s:  %s\n", "Parameter setting  ", "7 SAT");
		if (!c1_spec)
			c1 = 6.5;
		if (!c2_spec)
			c2 = 4;
		if (!c3_spec)
			c3 = 100000;
		if (!sp_spec)
			sp = 830;
	}
	if (c3 == 0) {
		printf("ERROR c3 = 0 not allowed!");
		exit(0);
	} else if ((sp < 0) || (sp > 1000)) {
		printf("ERROR ps is a probability -> 0<=sp<=1");
		exit(0);
	}
	if (randomCspec){
	if (randomC)
		pickClause = pickClauseRandom;
	else
		pickClause = pickClauseF;
	}else
		if (maxClauseSize!=minClauseSize){
			pickClause = pickClauseF;
			randomC = 0;
		}
}

int main(int argc, char *argv[]) {
	tryTime = 0.;
	parseParameters(argc, argv);
	setCpuLimit();
	parseFile();
	printFormulaProperties();
	setupSparrowParameters(); //call only after parsing file!!!
	//Initialize the look up table of Sparrow
	initSparrow();
	setupSignalHandler();
	printSolverParameters();
	srand(seed);

	allowedFlips = maxFlips;

	for (tryn = 0; tryn < maxTries; tryn++) {
		init();
		bestNumFalse = numClauses;
		if (use_luby) {
			maxFlips = luby() * luby_base;
			sp = sparray[tryn%num_sp];
			//fprintf(stderr, "c sp= %4d next restart after %4lld * %6d = %7lld flips\n", sp, maxFlips / luby_base, luby_base, maxFlips);
		}
		if (numFalse != 0) {
			for (flip = 1; flip <=maxFlips; flip++) {
				pickVar();
				flipAtom();
				varLastChange[bestVar] = flip;
				printStatsEndFlip(); //update bestNumFalse
				totalFlips++;
				if (totalFlips>=allowedFlips){
					maxTries=0;
					break;
				}
				if (numFalse == 0)
					break;
			}
		}
		tryTime = elapsed_seconds();
		totalTime += tryTime;
		if (numFalse == 0) {
			if (!checkAssignment()) {
				fprintf(stderr, "c ERROR the assignment is not valid!");
				printf("c UNKNOWN");
				return 0;
			} else {
				printEndStatistics();
				printf("s SATISFIABLE\n");
				if (printSol == 1)
					printSolution1line();
				return 10;
			}
		} else{;
			//fprintf(stderr,"c UNKNOWN best(%4d) current(%4d) (%-15.5fsec)\n", bestNumFalse, numFalse, tryTime);
		}
	}
	fprintf(stderr,"c UNKNOWN best(%4d) current(%4d) (%-15.5fsec)\n", bestNumFalse, numFalse, tryTime);
	printEndStatistics();
	if (maxTries > 1)
		fprintf(stderr, "c %-30s: %-8.3fsec\n", "Mean time per try", totalTime / (double) tryn);
	return 0;
}



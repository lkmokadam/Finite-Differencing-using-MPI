// lmokada Laxmikant Kishor Mokadam
/******************************************************************************
* FILE: p2.c
* DESCRIPTION:
*
* Users will supply the functions
* i.) fn(x) - the polynomial function to be analyized
* ii.) dfn(x) - the true derivative of the function
* iii.) degreefn() - the degree of the polynomial
*
* The function fn(x) should be a polynomial.
*
*
* LAST REVISED: 8/29/2017
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"


#define NGRID   1000 		// The number of grid points
#define XI	-1.0 		// first grid point
#define XF	1.5  		// last grid point
#define EPSILON	0.005	 	// the value of epsilon
#define DEGREE	3 		// the degree of the function fn() 
#define MIN_MAX 99           	// Tag for receiving MIN_MAX values
#define MIN_MAX_STATUS 100   	// Tag for receving status of min_max found by particular process
	
#define blocking_comm_flag 1    // if 0 uses non blocking communication else uses blocking communication 
#define manual_reduction_bit 1	// if 0 uses mpi_reduce reduction else uses manual reduction

//void print_function_data(int, double*, double*, double*);


/* Creating the err.dat file and write all the required data */
void print_error_data(int np, double, double, double*, double*, double*);

/* Implementation of sharing of values fn(x) across processes using blocking MPI_Send and MPI_Recv functions */
int blocking_comm(int * allocationArr , double * y );

/* Implementation of sharing of values fn(x) across processes using non-blocking MPI_Isend and MPI_Irecv functions */
int non_blocking_comm(int * allocationArr , double * y );

/* Implementation of calculation of minima and maxima by manually reducing the NGRID Communication accross the proceses is achived by MPI_Send and MPI_Recv functions*/
void manual_reduction(double * x, double * y, double * dy, double * aggregateErr, int dySize, double avg_err, double std_dev);

/* Implementation of calculation of minima and maxima by using mpi_reduce functions. This is achived by implementing custom operator using function mpi_reduce_get_min_max_op_fn */
void mpi_reduce_reduction(double * x, double * y, double * dy, double * aggregateErr, int dySize, double avg_err, double std_dev, int minMaxFoundFlag);


/*Function that is used to create custome MPI operator for MPI_Reduce function*/
void mpi_reduce_get_min_max_op_fn(void * local_min_max_arr, void *out_v_min_max, int *len, MPI_Datatype * dptr);

/* Struct that is used to achieve the exchange of min_max values using MPI_educe */
typedef struct{
	double x[DEGREE-1], count;
} min_max_arr;

int  main(int, char**);

/* Function implementation to get fn(x) */
double fn(double x)
{
	return pow(x, 3) - pow(x,2) - x + 1;
}

/* Function implementation to get the accurate dydx at value x */
double dfn(double x)
{
	return (3*pow(x,2)) - (2*x) - 1;
}



int main (int argc, char *argv[])
{
	int	i, j, index;  /* variable used for counter purpose */
	double	*x, *y, *dy;  /* array containers for values of gridpoints, fn(x) and dy/dx respectively */
	double dx ; 	      /* used for storing difference between subsequent gridpoints */
	double	*err , * aggregateErr, * bufErr; /* arrays for storing current rank's process' error , agrregateed error from all processes and buffer respectively */
	double *minMaxArr;     /* array for storing values of x at which local maxima/minima occurs */  
	double	avg_err, std_dev; /* used for storing average error and standard deviation respectively in root process */
	int	numproc, rank; /* used for storing number of processes and rank of current process */
	int	*allocationArr, *indexArr; /* arrays used for storing the number of grid points allocated to a particular process, and its global location */
	
	//variables used for local purposes
	int     remainder;     
	int 	dySize, countMinMax;
	int 	p_min_max_status, minMaxFoundFlag = 0;

	//initializing pointers to null
	x = dy = y = NULL;
	err = aggregateErr = bufErr = NULL;
	allocationArr = indexArr = NULL;
	minMaxArr = NULL; 	

	MPI_Init(&argc, &argv); /*Initializing the MPI environment*/
	MPI_Comm_size(MPI_COMM_WORLD, &numproc); /*Storing the number of processes*/
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /*Storing the rank of current process*/
	dx = (XF - XI)/(NGRID - 1);  				/* Calculates the value of dx as it is constant for given values of XF , XI and NGRID */ 

	allocationArr = (int *) malloc(sizeof(int) * numproc);  /* allocationArr[i] = number of grid points assigned to the process with Rank i */ 
	indexArr = (int *) malloc(sizeof(int) * numproc);


	/* Following code divides the NGRID among all process. 
	 * For Example, if NGRID is 5, NGRID with the boundry values is NGRID + 2 = 7 
	 *              If number of processers is 3 i.e. numproc = 3
	 * 		NGRID will be divided into the processes as 
	 * 			Rank 0 -> 3
	 *			Rank 1 -> 2
	 *			Rank 2 -> 2
	 */

									/********************************************************************************************************/
	for(i=0; i<numproc; i++)					/* Assignes the equal no of NGRID points to each process. i.e. if NGRID = 5, nproc = 3 ( as above ),	*/ 				
		allocationArr[i] = (NGRID+2)/numproc;			/* in this step, distribution of NGRID points will be : Rank 0 -> 2, Rank 1 -> 2, Rank 2 -> 2 		*/
									/********************************************************************************************************/
							  	   

	remainder = (NGRID+2) % numproc;				/* Finds the remaining number of NGRID points to distribute 						*/
	

									/********************************************************************************************************/
	for(i=0;remainder>0; i++,remainder--)				/* Distributes the remaining NGRID points to each process i.e. if NGRID = 5, nproc = 3 ( as above ),	*/ 
		allocationArr[i] += 1;			  		/* After this step, distribution of NGRID points will be : Rank 0 -> 3, Rank 1 -> 2, Rank 2 -> 2 	*/ 
									/********************************************************************************************************/
		
	/*  
	* indexArr is the array that contains information to merge the divided NGRID into a single NGRID 
	* Thus, it contains the global location of GRIDPOINT located at 0th location of divided NGRID Array
	* i.e. global location of GRID point = indexArr[Rank of process that has GRIDPOINT] + position of GRIDPOINT on divided NGRID array.
	*/

	indexArr[0] = 0;						/* 0th location of Rank 0th process's divided NGRID array will always have global location of 0 in 	*/ 
									/* whole NGRID array.											*/
	for(i=1; i<numproc; i++)					/* Calculating the subsequesnt Rank's 0th poisioned NGRID points global location 			*/
		indexArr[i] = indexArr[i-1] + allocationArr[i-1];	/* As per above example, indexArr will be [ 0, 3, 5]                             			*/ 

	x  =   (double*) malloc(allocationArr[rank] * sizeof(double));	/* Allocate the memory of array of values of x */

										
	i=0;											/********************************************************************************/
	if (rank == 0)										/*										*/
	{											/*										*/
		i=1;										/*										*/
		x[0] = XI - dx;									/*			Initializing the array of X values                      */
	}											/*										*/
												/*										*/
												/*										*/
	for (; i < allocationArr[rank] ; i++)							/*										*/
	{											/*										*/
		x[i] = XI + (XF - XI) * (double)(indexArr[rank] + i - 1)/(double)(NGRID - 1);	/*										*/
	}											/*										*/
												/********************************************************************************/
	

	if(rank==0 || rank==numproc-1)								/********************************************************************************/
		y  =   (double*) malloc((allocationArr[rank] + 1)* sizeof(double));		/*	Allocating the memory of y. For ran 0 and nproc-1, y size will be 	*/
	else											/*	rank+1 as it will receive only one vaalus across the boundry. But	*/
		y  =   (double*) malloc((allocationArr[rank] + 2)* sizeof(double));		/*	other ranks' process will receive +2 valus as they will receive values	*/
												/*	from rank-1 and rank+1 processes.					*/
												/********************************************************************************/
	

	if (rank == 0)										/********************************************************************************/
		for(i=0; i<allocationArr[rank]; i++ )						/*										*/
	  		y[i] = fn(x[i]);							/*										*/
	else											/*		Filling the values in y vector 					*/
		for(i=0; i<allocationArr[rank]; i++ )						/*										*/
	  		y[i+1] = fn(x[i]);							/*										*/
												/*										*/
												/********************************************************************************/
							
	#if blocking_comm_flag == 1 										
	blocking_comm(allocationArr , y);       						/* 	Blocking and non blocking communication function call		*/
	#else
	non_blocking_comm(allocationArr, y) ;
	#endif
												/*	to use non-blocking communication across processes				*/

	
	if(rank==0 || rank==numproc-1)							      	/********************************************************************************/	
		dySize = allocationArr[rank] - 1;						/*	calculating size of derivative array for current rank process 		*/
	else											/*										*/
		dySize = allocationArr[rank];							/********************************************************************************/
												
	dy =     (double*) malloc(dySize * sizeof(double));

	for (i = 0; i < dySize; i++)
		dy[i] = (y[i+2] - y[i])/(2.0 * dx);						/* 	Calculating the derivative of fn(X) at x 			*/		
	err = (double*)malloc(dySize * sizeof(double));

	if(rank == 0)										/********************************************************************************/
	{											/*										*/
		for (i = 0; i < dySize; i++)							/*  Calculating the error of each derivative found by finite diference method 	*/
			err[i] = fabs( dy[i] - dfn(x[i+1]) );					/*  The error is calculated by comparing it with actual derivative value 	*/
	}											/*										*/
	else											/*										*/
	{											/*										*/
		for (i = 0; i < dySize; i++)							/*										*/
			err[i] = fabs( dy[i] - dfn(x[i]) );					/*										*/
	}											/********************************************************************************/
	

	if(rank == 0)										/********************************************************************************/
	{											/*										*/
		aggregateErr = ( double * ) malloc ( NGRID * sizeof(double));			/*										*/
		bufErr = ( double * ) malloc ( NGRID * sizeof(double));				/*										*/
		index = 0;									/*										*/
		for(i=0; i<dySize; i++)								/*										*/
		{										/*										*/
			aggregateErr[index] = err[i];						/*										*/
			index++;								/*	All processes, except root process, send the array of errors calculated */
		}										/*	by each process repectively to the root	process			        */
	}											/*	The root process stores the received errors in an aggregateErr array	*/
												/*										*/
	if ( rank != 0 )									/*										*/
	{											/*										*/
		MPI_Send(err, dySize, MPI_DOUBLE, 0 ,0, MPI_COMM_WORLD);			/*										*/
	}											/*										*/
	else											/*										*/
	{											/*										*/
		for( i = 1 ; i < numproc; i++)							/*										*/
		{										/*										*/
			MPI_Recv(bufErr, NGRID, MPI_DOUBLE, i , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);		
			if( i == numproc-1)							/*										*/
			{									/*										*/
				for( j=0; j<allocationArr[i]-1; j++)				/*										*/
				{								/*										*/
					aggregateErr[index] = bufErr[j];			/*										*/
					index++;						/*										*/
				}								/*										*/
												/*										*/
			}									/*										*/
			else									/*										*/
			{									/*										*/
				for( j=0; j<allocationArr[i]; j++)				/*										*/
				{								/*										*/
					aggregateErr[index] = bufErr[j];			/*										*/
					index++;						/*										*/
				}			 					/*		  								*/	
			}									/*										*/
		} 										/*										*/
	}											/********************************************************************************/

	//root process calculates the average error and standard deviation													
	if(rank == 0)
	{
		avg_err = 0.0;

		for(i = 0; i < NGRID ; i++)
			avg_err += aggregateErr[i];

		avg_err /= (double)NGRID;

		std_dev = 0.0;

		for(i = 0; i< NGRID; i++)
			std_dev += pow(aggregateErr[i] - avg_err, 2);

		std_dev = sqrt(std_dev/(double)NGRID); 	  	
	
	}

	//All processes compares the derivative calculated by them with the epsilon
	//and for those dervatives whose values are less than epsilon their corresponding values of x are sent to root process  
	

	//Root process receives all local maxima and minima and processes them
	//If number of local maxima/minima received is more than DEGREE-1 it prints an error

	#if manual_reduction_bit == 1 										
	manual_reduction(x,y,dy,aggregateErr,dySize,avg_err, std_dev);      						/* 	manual reduction and mpi_reduce_reduction function call */
	#else
	mpi_reduce_reduction(x,y,dy,aggregateErr,dySize,avg_err, std_dev,minMaxFoundFlag);
	#endif
	
	
	

	MPI_Finalize();
	
	//free all dynamically allocated arrays
	free(y);
	free(dy);
	free(err);
	free(x);
	free(aggregateErr);
	free(bufErr);
	free(allocationArr);
	free(indexArr);
	free(minMaxArr);

	return 0;
}


/*Function Name: blocking_comm*/
/*Purpose: This function performs communication across the boundaries.*/
/*This helps process to know the value of function at left and right */
/*edge of the decomposed grid. The communication done is blocking*/
/*communication by using MPI_Send and MPI_Recv*/

int blocking_comm(int * allocationArr , double * y )
{
	int rank,numproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	if(rank==0)
	{
		//Since process with rank '0' has first part of decomposed grid,
		//so it sends boundary value on its right side to process with rank '1' only  
		MPI_Send(&y[allocationArr[rank]-1], 1, MPI_DOUBLE, rank+1,   0, MPI_COMM_WORLD);

		//Process with rank '0' need to know boundary value only on its right side,
		//provided to it by process of rank 1
		MPI_Recv(&y[allocationArr[rank]], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else if(rank==numproc-1)
	{ 
		//Since process with last rank has last part of decomposed grid,
		//so it sends boundary value on its left side to second last process 
		MPI_Send(&y[1], 1, MPI_DOUBLE, rank-1,   0, MPI_COMM_WORLD);

		//Process with last rank need to know boundary value on its left side,
		//provided to it by second last process
		MPI_Recv(&y[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		//All those processes which are neither first nor last need to communicate the values
		//at its right and left boundaries to its predecessor and successor processes.
		MPI_Send(&y[1], 1, MPI_DOUBLE, rank-1,   0, MPI_COMM_WORLD);
		MPI_Send(&y[allocationArr[rank]], 1, MPI_DOUBLE, rank+1,   0, MPI_COMM_WORLD);

		//All those processes which are neither first nor last shall receive boundary values
		//from its predecessor and successor processes		
		MPI_Recv(&y[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&y[allocationArr[rank]+1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
}

/*Function Name: non_blocking_comm*/
/*Purpose: This function performs communication across the boundaries.*/
/*This helps process to know the value of function at left and right */
/*edge of the decomposed grid. The communication done is non-blocking*/
/*communication by using MPI_Isend and MPI_Irecv*/
int non_blocking_comm(int * allocationArr , double * y )
{

	int rank,numproc;
	MPI_Request request_s, request_r, request_s_1, request_r_1;
	MPI_Status status_s, status_r,status_s_1, status_r_1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	if(rank==0)
	{
		//Since process with rank '0' has first part of decomposed grid,
		//so it sends boundary value on its right side to process with rank '1' only  
		MPI_Isend(&y[allocationArr[rank]-1], 1, MPI_DOUBLE, rank+1,   0, MPI_COMM_WORLD, &request_s);

		//Process with rank '0' need to know boundary value only on its right side,
		//provided to it by process of rank 1
		MPI_Irecv(&y[allocationArr[rank]], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request_r);
		
		//Calculation of derivative, i.e dy/dx, cannot proceed until boundary values are received
		MPI_Wait(&request_r, &status_r);
	}
	else if(rank==numproc-1)
	{
		//Since process with last rank has last part of decomposed grid,
		//so it sends boundary value on its left side to second last process   
		MPI_Isend(&y[1], 1, MPI_DOUBLE, rank-1,   0, MPI_COMM_WORLD, &request_s);

		//Process with last rank need to know boundary value on its left side,
		//provided to it by second last process
		MPI_Irecv(&y[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request_r);

		//Calculation of derivative, i.e dy/dx, cannot proceed until boundary values are received
		MPI_Wait(&request_r, &status_r);
	}
	else
	{
		//All those processes which are neither first nor last need to communicate the values
		//at its right and left boundaries to its predecessor and successor processes.
		MPI_Isend(&y[1], 1, MPI_DOUBLE, rank-1,   0, MPI_COMM_WORLD,&request_s);
		MPI_Isend(&y[allocationArr[rank]], 1, MPI_DOUBLE, rank+1,   0, MPI_COMM_WORLD,&request_s_1);

		//All those processes which are neither first nor last shall receive boundary values
		//from its predecessor and successor processes		
		MPI_Irecv(&y[0], 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &request_r);
		MPI_Irecv(&y[allocationArr[rank]+1], 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &request_r_1);

		//Calculation of derivative, i.e dy/dx, cannot proceed until boundary values are received		
		MPI_Wait(&request_r, &status_r);
		MPI_Wait(&request_r_1, &status_r_1);
	}
}

/*Function Name: print_error_data*/
/*Purpose: The function writes value of average error, standard deviation, value of x where local minima & maxima*/
/*of function occurs and value of function at these points in err.dat file. Further, the function writes */
/*the value of x and the corresponding value of error, i.e. the difference between the calculated derivative and */
/*the actual value of derivative at point x, in err.dat file*/
void print_error_data(int np, double avgerr, double stdd, double *x, double *err, double *local_min_max)
{
	int   i;
	FILE *fp = fopen("err.dat", "w");

	fprintf(fp, "%e\n%e\n", avgerr, stdd);

	for(i = 0; i<DEGREE-1; i++)
	{
		if (local_min_max[i] != INT_MAX)
			fprintf(fp, "(%f, %f)\n", local_min_max[i], fn(local_min_max[i]));
		else
			fprintf(fp, "(UNDEF, UNDEF)\n");
	}	
	
	double x_diff = x[1] - x[0];

	for(i = 0; i < np; i++)
	{
		fprintf(fp, "%f %e \n", x[0]+i*x_diff, err[i]);
	}
	fclose(fp);
}


/*Function name: manual_reduction															*/
/*Purpose: The function calculated the minima and maxima using manual reduction. 									*/
/*The steps are as follows:																*/
/*	1. If process finds a local minima or maxima, it first sends the minMaxFoundTag with number of minima_maxima found by processto root 		*/
/*	2. After getting the message with tag 'minMaxFoundTag', Root executes the receive function with the minMaxFoundTag snder as source for		*/ 
/*	number of times specified by message in data. 													*/
/*	3. Repeat steps 1-2 for each process	 													*/ 


void manual_reduction(double * x, double * y, double * dy, double * aggregateErr, int dySize, double avg_err, double std_dev)
{

	int rank,numproc;
	int countMinMax,i,j, minMaxFoundFlag = 0;
	double * minMaxArr;
	int p_min_max_status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	

	if(rank != 0)											/**************************************************************************/
	{												/* If non-root process find minima_maxima, it first sends the number of	  */
		for (i = 0; i < dySize; i++)								/* min_max found ( say n) to root process and then sends the x values with*/
													/* n number of send function. O pother hand, root process receives  */
		{											/* MIN_MAX_STATUS flagged message with number n. It then runs receive	*/
			if(fabs(dy[i]) < EPSILON)							/* function n number of times, and collects the x values at which min_max*/
			{										/* exists								*/
				MPI_Send(&x[i], 1, MPI_DOUBLE, 0 , MIN_MAX, MPI_COMM_WORLD);		/*************************************************************************/
				minMaxFoundFlag++;
			}
		}

		MPI_Send(&minMaxFoundFlag, 1, MPI_INT, 0 , MIN_MAX_STATUS, MPI_COMM_WORLD);

	}
	if(rank == 0)
	{
		countMinMax = 0;
		minMaxArr = ( double * ) malloc ( (DEGREE-1) * sizeof(double));

		for(i=0; i<DEGREE-1; i++)
	  	{
			minMaxArr[i]=INT_MAX; 							
	  	}

		for (i = 0; i < dySize; i++)
		{
			if(fabs(dy[i]) < EPSILON)
			{
				minMaxArr[countMinMax] = x[i+1];
				countMinMax++;
			}	
		}
	
		for(i=1; i<numproc; i++)
		{
		
			MPI_Recv(&p_min_max_status, 1, MPI_INT, i , MIN_MAX_STATUS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (p_min_max_status > 0)
				for(j=0;j<p_min_max_status;j++)
				{

					if(countMinMax >= DEGREE-1)
					{
						printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
						printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
						printf("Reseting count to zero.\n");
						countMinMax = 0;
				
					}
					MPI_Recv(&minMaxArr[countMinMax], 1, MPI_DOUBLE, i , MIN_MAX , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					countMinMax++;

				
		      		}
		
		}

		print_error_data(NGRID, avg_err, std_dev, &x[1], aggregateErr, minMaxArr); //call function that prints data to file
	}
}



/*Function Name: mpi_reduce_get_min_max_op_fn*/
/*Purpose: The function gets the local minima maxima of the processes and sends accumulate in the array. and send it to the next */
/*level. To encapsulate the data to be send by MPI_reduce, a struct min_max_arr is used. */


void mpi_reduce_get_min_max_op_fn(void * local_min_max_arr, void *out_v_min_max, int *len, MPI_Datatype * dptr)
{
	
	int rank ; min_max_arr temp ; 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	min_max_arr * in_min_max_arr;
	in_min_max_arr = ( min_max_arr *) local_min_max_arr;
	min_max_arr * out_min_max = ( min_max_arr * ) out_v_min_max; int i = 0;

	for (i = 0; i < DEGREE-1; i++)
	{
		if (out_min_max->count > (DEGREE-1))
		{
			printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
			printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
			printf("Reseting count to zero.\n");
			out_min_max->count =0;
		}
		if(in_min_max_arr->x[i] == INT_MAX) continue;	
		out_min_max->x[(int)out_min_max->count] = in_min_max_arr->x[i];
		out_min_max->count++;	
	}
}


/* Function Name: mpi_reduce_reduction */
/* Purpose: The function calculated the minima and maxima using mpi_reduce reduction. The function calculate the local minima and maxima and send the data to next  */
/* level in the communication hierarchy of mopi_reduce. On each node it accumulate data from various node until the data is reached to the root process  */
/* the function access the dy values of x values. check wheather it is less than Espilon. If yes, it then accumulate the values and send to the MPI_Reduce. */
/* MPI_Reduce function performs the operation as explained for function mpi_reduce_get_min_max_op_fn. and send to the next level / root. */


void mpi_reduce_reduction(double * x, double * y, double * dy, double * aggregateErr, int dySize, double avg_err, double std_dev, int minMaxFoundFlag)
{

	int rank,numproc;
	int countMinMax,i,j;
	double * minMaxArr;
	int p_min_max_status;

	min_max_arr global_min_max;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);

	MPI_Op min_max_op;
	MPI_Datatype ctype;

	MPI_Type_contiguous(DEGREE-1*+1,MPI_DOUBLE,&ctype);
	MPI_Type_commit(&ctype);

	countMinMax = 0;
	minMaxArr = ( double * ) malloc ( (DEGREE-1) * sizeof(double));

	for(i=0; i<DEGREE-1; i++)
  	{
		minMaxArr[i]=INT_MAX; 
  	}

	for (i = 0; i < dySize; i++)
	{
		if(fabs(dy[i]) < EPSILON)
		{
			if ( rank == 0 )
				minMaxArr[countMinMax] = x[i+1];
			else
				minMaxArr[countMinMax] = x[i];
			countMinMax++;
		}	
	}
	min_max_arr local_min_max_arr;

	//Initializes local_min_max_arr with INT_MAX
	for( i = 0 ; i < (DEGREE -1); i++)
	{
		local_min_max_arr.x[i] = INT_MAX;
	}
	
	//Fills up values in local_min_max_arr
	for( i = 0; i < countMinMax ; i++)
	{
		local_min_max_arr.x[i] = minMaxArr[i];
	}

	local_min_max_arr.count = countMinMax;
	
	// Initializes output global_min_max variable with INT_MAX
	for( i = 0 ; i < DEGREE-1; i++)
	{
		global_min_max.x[i] = INT_MAX;
	}
	global_min_max.count = 0;
	
	MPI_Op_create(mpi_reduce_get_min_max_op_fn,1,&min_max_op);

	if(rank==0)
	{
		for (i = 0; i < DEGREE-1; i++)
		{
			if (global_min_max.count > (DEGREE-1))
			{
				printf("Warning: You have detected more than the maximum possible local minima/maxima.\n");
				printf("Ensure that DEGREE is accurate or reduce your EPSILON.\n");
				printf("Reseting count to zero.\n");
				global_min_max.count =0;
			}
			if(local_min_max_arr.x[i] == INT_MAX) continue;	
			global_min_max.x[(int)local_min_max_arr.count] = local_min_max_arr.x[i];
			global_min_max.count++;	
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Reduce(&local_min_max_arr, &global_min_max, 1, ctype, min_max_op, 0, MPI_COMM_WORLD);
		
	if(rank == 0)
	{
		print_error_data(NGRID, avg_err, std_dev, &x[1], aggregateErr, global_min_max.x); //call function that prints data to file
	}
}

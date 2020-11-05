#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <SFMT.h>
#include <omp.h>
#include "mnist.c"

#define TRAINING_DATA_SIZE MNIST_TRAINING_DATA_SIZE
#define DEBUG 1
#define PREPROCESSING 0
#define L2REGULARIZARION 1
#define ETA 0.1
#define IMAGE 0

extern void sfmt_init_gen_rand(sfmt_t * sfmt, uint32_t seed);
extern double sfmt_genrand_real2(sfmt_t * sfmt);
extern double sfmt_genrand_real3(sfmt_t * sfmt);

typedef double Neuron, Delta, Weight;
typedef struct { Weight *w; Weight *dw; int n_pre; int n_post; } Connection;
typedef struct { Neuron *z; Delta *delta; int n; } Layer;
typedef struct { Layer *layer; Connection *connection; sfmt_t rng; int n; } Network;

double all_to_all ( Network *n, const int i, const int j, const int n_pre ) { return 1.; }
double uniform_random ( Network *n, const int i, const int j, const int n_pre ) { return 1.-2.*sfmt_genrand_real2 ( &n -> rng ); }
double gaussian_random ( Network *n, const int i, const int j, const int n_pre ) {
	double z;
	z = sqrt( -2.0 * log( sfmt_genrand_real3 ( &n -> rng ) )) * sin(2.0*M_PI*sfmt_genrand_real3 ( &n -> rng ));
	return z/n_pre;
}
double sparse_random ( Network *n, const int i, const int j, const int n_pre )
{
  return ( sfmt_genrand_real2 ( &n -> rng ) < 0.5 ) ? uniform_random ( n, i, j, n_pre ) : 0.;
}
double sigmoid ( double x ) { return 1. / ( 1. + exp ( - x ) ); }
double diff_sigmoid ( double z ) { return z * ( 1. - z ); } // f'(x) = f(x) ( 1 - f(x) ) = z ( 1 - z )

double ReLU (double x) {return (x>0)?x:0.;}
double diff_ReLU(double z) { return (z>0)?1:0.; }

double LReLU (double x) {return (x>0)?x:x*0.01;}
double diff_LReLU(double z) { return (z>0)?1:0.01; }

void createNetwork ( Network *network, const int number_of_layers, const sfmt_t rng )
{
  network -> layer = ( Layer * ) malloc ( number_of_layers * sizeof ( Layer ) );
  network -> connection = ( Connection * ) malloc ( number_of_layers * sizeof ( Connection ) );
  network -> n = number_of_layers;
  network -> rng = rng;
}

void deleteNetwork ( Network *network )
{
  free ( network -> layer );
  free ( network -> connection );
}

void createLayer ( Network *network, const int layer_id, const int number_of_neurons )
{
  Layer *layer = &network -> layer [ layer_id ];

  layer -> n = number_of_neurons;

  int bias = ( layer_id < network -> n - 1 ) ? 1 : 0; // 出力層以外はバイアスを用意

  layer -> z = ( Neuron * ) malloc ( ( number_of_neurons + bias ) * sizeof ( Neuron ) );
  for ( int i = 0; i < layer -> n; i++) { layer -> z [ i ] = 0.; } // 初期化
  if ( bias ) { layer -> z [ layer -> n ] = +1.; } // バイアス初期化

  // Deltaを追加
  layer -> delta = ( Delta * ) malloc ( ( number_of_neurons + bias ) * sizeof ( Delta ) );
  for ( int i = 0; i < layer -> n; i++) { layer -> delta [ i ] = 0.; }
  if ( bias ) { layer -> delta [ layer -> n ] = 0.; } // バイアス初期化

}

void deleteLayer ( Network *network, const int layer_id )
{
  Layer *layer = &network -> layer [ layer_id ];
  free ( layer -> z );
  free ( layer -> delta );
}

void createConnection ( Network *network, const int layer_id, double ( *func ) ( Network *, const int, const int, const int ) )
{
  Connection *connection = &network -> connection [ layer_id ];

  const int n_pre = network -> layer [ layer_id ] . n + 1; // +1 for bias
  const int n_post = ( layer_id == network -> n - 1 ) ? 1 : network -> layer [ layer_id + 1 ] . n;

  connection -> w = ( Weight * ) malloc ( n_pre * n_post * sizeof ( Weight ) );
  for ( int i = 0; i < n_post; i++ ) {
    for ( int j = 0; j < n_pre; j++ ) {
      connection -> w [ j + n_pre * i ] = func ( network, i, j, n_pre );
    }
  }

  connection -> dw = ( Weight * ) malloc ( n_pre * n_post * sizeof ( Weight ) );
  for ( int i = 0; i < n_post; i++ ) {
    for ( int j = 0; j < n_pre; j++ ) {
      connection -> dw [ j + n_pre * i ] = 0.;
    }
  }

  connection -> n_pre = n_pre;
  connection -> n_post = n_post;
}

void deleteConnection ( Network *network, const int layer_id )
{
  Connection *connection = &network -> connection [ layer_id ];
  free ( connection -> w );
  free ( connection -> dw );
}

void setInput ( Network *network, Neuron x [ ] )
{
  Layer *input_layer = &network -> layer [ 0 ];
  for ( int i = 0; i < input_layer -> n; i++ ) {
    input_layer -> z [ i ] = x [ i ];
  }
}

void forwardPropagation ( Network *network, double ( *activation ) ( double ) )
{
  for ( int i = 0; i < network -> n - 1; i++ ) {
    Layer *l_pre = &network -> layer [ i ];
    Layer *l_post = &network -> layer [ i + 1 ];
    Connection *c = &network -> connection [ i ];
    for ( int j = 0; j < c -> n_post; j++ ) {
      Neuron u = 0.;
      for ( int k = 0; k < c -> n_pre; k++ ) {
	u += ( c -> w [ k + c -> n_pre * j ] ) * ( l_pre -> z [ k ] );
      }
      l_post -> z [ j ] = activation ( u );
    }
  }
}

double updateByBackPropagation ( Network *network, Neuron z [ ], double ( *diff_activation ) ( double ), const double Eta )
{
	double err = 0.;
  
	Layer *l = &network->layer[ network->n - 1 ];

  //Error
	for(int k = 0; k < l->n; k++){
		err += 0.5*((l->z[k] - z[k])*(l->z[k] - z[k]));
	}

	//OutputDelta
	for(int i = 0; i < l->n;i++){
	  Neuron o = l->z[i];
    	  l -> delta[i] = (z[i] - o);
	}

	for(int i = network->n - 2; i >= 0; i--){
		Connection *c = &network->connection[i];
    		Layer *l_post = &network->layer[ i + 1 ];
		Layer *l  = &network->layer[ i ];

		//dw
		for(int j = 0; j < c -> n_post; j++){
			for(int k = 0; k < c -> n_pre; k++){
			    c->dw[k + c->n_pre*j] += Eta*l_post->delta[j] * diff_activation(l_post->z[j]) * l->z[k];
			}
		}

		//InitDelta
	  	for(int j = 0; j < l->n;j++) l->delta[j] = 0.;

		//Delta
	  	for(int j = 0; j < c->n_pre; j++){
			for(int k = 0; k < c->n_post; k++){
        			Layer *l  = &network->layer[ i ];
				l->delta[j] += l_post->delta[k]*diff_activation(l_post->z[k])*c->w[ j + c->n_pre*k];
		  	}
	  	}
	}

	return err;
}

void initializeDW ( Network *network )
{
  for ( int layer_id = 0; layer_id < network -> n - 1; layer_id++ ) {
    Connection *c = &network -> connection [ layer_id ];
    for ( int i = 0; i < c -> n_post; i++ ) {
      for ( int j = 0; j < c -> n_pre; j++ ) {
	c -> dw [ j + c -> n_pre * i ] = 0.;
      }
    }
  }
}

void updateW ( Network *network, int mini_batch_size, const double eta, const double lambda )
{
  for ( int layer_id = 0; layer_id < network -> n - 1; layer_id++ ) {
    Connection *c = &network -> connection [ layer_id ];
    for ( int i = 0; i < c -> n_post; i++ ) {
      for ( int j = 0; j < c -> n_pre; j++ ) {
					if(L2REGULARIZARION)c->w[j+c->n_pre*i] *= 1 - eta*lambda/mini_batch_size;
	c -> w [ j + c -> n_pre * i ] += c -> dw [ j + c -> n_pre * i ]/mini_batch_size ;
      }
    }
  }
}

int* order_initialize(){
	int *order = (int *)malloc(TRAINING_DATA_SIZE*sizeof(int));
	for(int i = 0; i < TRAINING_DATA_SIZE; i++){
		order[i] = i;
	}
	return order;
}

void order_shuffle(int *order, sfmt_t rng){
	int tmp, rand;
	for(int i = TRAINING_DATA_SIZE - 1; i > 0; i--){
		rand = i * sfmt_genrand_real2(&rng);
		tmp = order[rand];
		order[rand] = order[i];
		order[i] = tmp;
	}
}

void order_reset(int *order){
	for(int i = 0; i < TRAINING_DATA_SIZE; i++){
		order[i] = i;
	}
}

void preprocessing( double **data, int N){
	double mu [MNIST_IMAGE_SIZE] = {0.};
	double sigma [MNIST_IMAGE_SIZE] = {0.};

	for(int i = 0; i < N; i++){
		for(int j = 0; j < MNIST_IMAGE_SIZE;j++){
			mu[j] += data[i][j];
		}
	}

	for(int j = 0; j < MNIST_IMAGE_SIZE;j++){
		mu[j] = mu[j]/(double)N;
	}

	for(int i = 0; i < N; i++){
		for(int j = 0; j < MNIST_IMAGE_SIZE;j++){
			sigma[j] += (data[i][j] - mu[j])*(data[i][j] - mu[j]);
		}
	}
	for(int j = 0; j < MNIST_IMAGE_SIZE;j++){
		sigma[j] = sqrt(sigma[j]/(double)N);
	}

	for(int i = 0; i < N; i++){
		for(int j = 0; j < MNIST_IMAGE_SIZE; j++){
			data[i][j] = data[i][j] - mu[j];
			if(sigma[j]!=0) data[i][j] /=sigma[j];
		}
	}
	return;
}


void free_2dim_array(double **pointer, int N){
	for(int i = 0; i < N; i++){
		 free(pointer[i]);
	}
	free(pointer);
}

void weight_visualization(Network *network, const int pre_layer, const int neuron_id ,const char *filename){
	Connection *c = &network->connection[pre_layer];
	if(neuron_id >= c->n_post){
	  fprintf(stderr, "The neuron %d does not exist.\n", neuron_id);
	}
  gdImagePtr im = gdImageCreate( MNIST_IMAGE_ROW_SIZE, MNIST_IMAGE_COL_SIZE );
	
	const int n_grayscale = 256;
	int gray[n_grayscale];
	for ( int i = 0; i < n_grayscale; i++ ) { gray [ i ] = gdImageColorAllocate ( im, i, i, i ); }

	int max = 0;
	for(int i=0; i < MNIST_IMAGE_SIZE;i++){
		max = (c->w[c->n_pre*neuron_id + i] > max)?c->w[c->n_pre*neuron_id]:max;
	}

	for(int i = 0; i < MNIST_IMAGE_ROW_SIZE; i++){
		for(int j = 0; j < MNIST_IMAGE_COL_SIZE;j++){
			double weight = c->w[ j + MNIST_IMAGE_COL_SIZE * i + c->n_pre * neuron_id ]/max;
			if(weight < 0) weight = 0;
		  int index = (int) ( ( n_grayscale - 1 ) * weight );
			gdImageSetPixel ( im, j, i, gray [ index ] );
		}
	}

	{
		FILE *file = fopen(filename, "wb");
		gdImagePng ( im, file );
		fclose (file);
	}

	gdImageDestroy(im);

  return;
}

void dump( Network *network )
{
	// network 
	fprintf(stdout,"Network Info\n");
	fprintf(stdout,"  layers:\t%d\n", network->n);
	// layer
	for(int  i = 0; i < network->n; i++){
		Layer *l = &network->layer[i];
		fprintf(stdout,"Layer  %d\n",i);
		fprintf(stdout,"  neurons:\t%d\n",l->n);
	// neuron
		for(int j = 0; j < l->n;j++){
			fprintf(stdout,"    neuron  %d:\t %lf \n",j,l->z[j]);
		}
	}
	// connection
	for(int i = 0; i < network->n - 1; i++){
		Connection *c = &network->connection[i];
		fprintf(stdout,"Connection %d -> %d\n", i, i+1);
		fprintf(stdout,"  weight\n");
		for(int j = 0; j < c->n_pre; j++){
			if(j == c->n_pre-1) fprintf(stdout,"   bias\n");
			for(int k = 0; k < c->n_post;k++){
				fprintf(stdout,"    %d->%d: %lf\n",j,k,c->w[c->n_pre*k+j]);
			}
		}
	}
}

int main ( int argc, char **argv )
{
	double **training_image, **test_image;
	int *training_label, *test_label;
	int *order = NULL;
	order = order_initialize ();
	mnist_initialize ( &training_image, &training_label, &test_image, &test_label);

  sfmt_t rng;
  sfmt_init_gen_rand ( &rng, getpid ( ) );

	int epochs;
	int mini_batch_size;
	double Eta,lambda;
	int middle_layer_neurons;
	epochs = (argc >= 2)?atoi(argv[1]):1;
	mini_batch_size = (argc >= 3)?atoi(argv[2]):1;
	middle_layer_neurons = (argc >= 4)?atoi(argv[3]):32;
	Eta = (argc >= 5)?atof(argv[4]):ETA;
	lambda = (argc >= 6)?atof(argv[5]):0.1;

  Network network;
  createNetwork ( &network, 3, rng );
  createLayer ( &network, 0, MNIST_IMAGE_SIZE );
  createLayer ( &network, 1, middle_layer_neurons );
  createLayer ( &network, 2, MNIST_LABEL_SIZE );
  createConnection ( &network, 0, gaussian_random );
  createConnection ( &network, 1, gaussian_random );


			order_reset(order);	

 fprintf(stderr,"start training\n");
  // Training
	int i = 0;
	int e = 0;
	if(PREPROCESSING){
		preprocessing( training_image, TRAINING_DATA_SIZE );
		preprocessing(test_image,MNIST_TEST_DATA_SIZE);
		fprintf(stderr,"preprocessed\n");
	}
	for(e = 0; e < epochs; e++){
		order_shuffle(order, rng);
  	for(i = 0;i < TRAINING_DATA_SIZE / mini_batch_size;i++){
    	initializeDW ( &network );
			int j;
			double error = 0.;
    	for ( j = 0; j < mini_batch_size; j++ ) {
      	int k = order[i*mini_batch_size+j];
      	setInput ( &network, training_image [ k ] );
      	forwardPropagation ( &network, ReLU);
				double z [MNIST_LABEL_SIZE] = {0};
				z[training_label[k]] = 1.;
      	error += updateByBackPropagation ( &network, z, diff_ReLU, Eta );
    	}
    	fprintf ( stderr,"\r%d / %d :  %d / %d ", e+1,epochs, i*mini_batch_size+j, TRAINING_DATA_SIZE );
   		updateW ( &network, mini_batch_size, Eta, lambda );
		}

		// Test after epoch
		if(DEBUG){
			Layer *output_layer = &network . layer [ network. n - 1 ];
  		const int n = output_layer -> n;
			int correct = 0;
  		for ( int i = 0; i < TRAINING_DATA_SIZE; i++ ) {
    		setInput ( &network, training_image[i] );
    		forwardPropagation ( &network, ReLU );
    		//dump ( &network );
				int maxj = 0;
				double maxz = 0.;
   	 		for ( int j = 0; j < n; j++ ) {
					if (output_layer -> z[j] > maxz) {maxz = output_layer->z[j];maxj=j;}
    		}
				correct += (maxj == training_label[i]);
  		}
			fprintf( stderr, " : %lf" ,(double)correct / TRAINING_DATA_SIZE);
		}
		if(DEBUG){
			Layer *output_layer = &network . layer [ network. n - 1 ];
  		const int n = output_layer -> n;
			int correct = 0;
  		for ( int i = 0; i < MNIST_TEST_DATA_SIZE; i++ ) {
    		setInput ( &network, test_image[i] );
    		forwardPropagation ( &network, ReLU );
    		//dump ( &network );
				int maxj = 0;
				double maxz = 0.;
   	 		for ( int j = 0; j < n; j++ ) {
					if (output_layer -> z[j] > maxz) {maxz = output_layer->z[j];maxj=j;}
    		}
				correct += (maxj == test_label[i]);
  		}
			fprintf( stderr, " : %lf" ,(double)correct / MNIST_TEST_DATA_SIZE);
		}

		fprintf(stderr, "\n");
	}

  // Evaluatation
	fprintf(stderr, "\rstart evaluation\n");
  Layer *output_layer = &network . layer [ network. n - 1 ];
  const int n = output_layer -> n;
	int correct = 0;
  for ( int i = 0; i < MNIST_TEST_DATA_SIZE; i++ ) {
    setInput ( &network, test_image[i] );
    forwardPropagation ( &network, ReLU );//sigmoid,ReLU, LReLU
    //dump ( &network );
		int maxj = 0;
		double maxz = 0.;
    for ( int j = 0; j < n; j++ ) {
			if (output_layer -> z[j] > maxz) {maxz = output_layer->z[j];maxj=j;}
    }
		correct += (maxj == test_label[i]);
  }
	fprintf(stderr, "done\n");
	fprintf(stdout, "%d\t%d\t%f\n",epochs,mini_batch_size,(double) correct / MNIST_TEST_DATA_SIZE);

  //dump ( &network );
	//Visualiz
	if(IMAGE){
		for(int i = 0; i< network.connection[0].n_post; i++){
			char fn[1024];
			sprintf(fn, "./data_bp_mnist/fashion/pure_%d.png",i);
			weight_visualization(&network,0,i,fn);
		}
	}

  deleteConnection ( &network, 1 );
  deleteConnection ( &network, 0 );
  deleteLayer ( &network, 2 );
  deleteLayer ( &network, 1 );
  deleteLayer ( &network, 0 );
  deleteNetwork ( &network );

	mnist_finalize(training_image, training_label, test_image, test_label);

  return 0;
}

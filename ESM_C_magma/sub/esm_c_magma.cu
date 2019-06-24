//

// the way to code the magma 
// 1. define all the objects in the CUDA and magma
// 2. allocate GPU memory for the matrix (cudaMalloc)
// 3. copy the matrix to the the GPU
// 4. use diagonalization function in magma
// 5. copy the result from GPU to Memory
// 6. delete CUDA & magma objects and GPU memory space

#include "esm_c_magma.h" 


// Global variable to catch interrupt and terminate signals
volatile sig_atomic_t interrupted=false;


//use input name to get output name
string FileName(string name)
{
    int site = name.find_last_of('.');
    if (site == 0)
    {
        name = "output";
    }
    else if (site > 0)
    {
        name.erase(site, name.size() - site + 1);
    }
    return name;
}


//host name function
string HostName()
{
    string hostname;
    //for linux system
    ifstream hostname_file("/etc/hostname", ifstream::in);
    hostname_file >> hostname;
    return hostname;
}

// void mkdir(string dir)
// {
    // ofstream file("mkdir.bat");
    // file << "@echo off" << endl;
    // file << "if not exist " << dir << " mkdir " << dir << endl;
    // file << "chdir " << dir << endl;
    // file.close();

    // system("mkdir.bat");

    // remove("mkdir.bat");

    // cout << "**************************************************************" << endl;
    // cout << "entering ./" << dir << endl;
// }

int get_mon_state(int sysnx, int n) {
    int temp = (sysnx - 1) / pow(mon_kount, (n - 1));
    int get_mon = temp % mon_kount + 1;
    return get_mon;
}

int get_numvib(int molecule_state[]) {
    int numvib = 0;
    int n;
    for (n = 0; n < nmax; ++n) {
        numvib = numvib + mon_state[molecule_state[n] - 1].vib1 +
            mon_state[molecule_state[n] - 1].vib2;
    }

    return numvib;
}

double get_distance(int n1, int da1, int n2, int da2) {
    double distance = 0.0;
    int i = 9*n1 + 3*da1 - 12;
    int j = 9*n2 + 3*da2 - 12;
    distance = pow((mol1pos[i]-mol1pos[j]),2)
                  +pow((mol1pos[i+1]-mol1pos[j+1]),2)
                  +pow((mol1pos[i+2]-mol1pos[j+2]),2); 
    distance = sqrt(distance);

    return distance;
}

int get_charge(int state, int da){
    int charge = 0; 

    if ( state == es_z1 ){
        if ( da == leftdonor ){
            charge = 1;
        } else if ( da == acceptor ){
            charge = -1;
        }
    } else if( state == es_z2 ){
        if ( da == rightdonor ){
            charge = 1;
        } else if ( da == acceptor ){
            charge = -1;
        }
    }

    return charge; 
}


double factorial(int n) {
    double factorial_1 = 1.0;
    if (n < 0) {
        cout << "Factorial not calculatable for: " << n << endl;
        exit(0);
    }
    else {
        if (n != 0) {
            for (int i = 2; i <= n; i++) {
                factorial_1 = factorial_1 * i;
            }
        }
    }
    return factorial_1;
}

double volap(double lambda1, int vib1, double lambda2, int vib2) {
    double lambda = lambda2 - lambda1;
    double volap_1 = 0.0;
    for (int k = 0; k <= min(vib1, vib2); k++){
    volap_1 = volap_1 + pow((-1.0), (vib2 - k)) /
        (factorial(vib1 - k)*factorial(k)*
        factorial(vib2 - k))*
        pow(lambda, (vib1 + vib2 - 2 * k));
    }

    volap_1 = volap_1 * sqrt(1.0*factorial(vib1)*
        factorial(vib2))*exp(-1.0* pow(lambda, 2) / 2.0);

    if (volap_1 != volap_1) {
        cout << "Volap Error:: volap: " << volap_1 << endl;
        cout << "Aborting " << volap_1 << endl;
        exit(0);
    }
    return volap_1; 
}

// void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
    // MKL_INT i, j;
    // printf( "\n %s\n", desc );
    // for( i = 0; i < m; ++i ) {
        // for( j = 0; j < n; ++j ) printf( " %6.2lf", a[i*lda+j] );
        // printf( "\n" );
    // }
// }

int main(int argc, char** argv) {

    time_t start=time(NULL), end;

    magma_print_environment();

    // GPU variables                 
    //const int           blockSize = 128;        // The number of threads to launch per block


    // ***              Variable Declaration            *** //
    // **************************************************** //

    //input file check
    if ( argc != 2 ){
        printf("Usage:\n"
               "\tInclude as the first argument either the name of an input file,  or a checkpoint\n"
               "\tfile with extension '.cpt' if restarting the calculation. No other arguments are\n"
               "\tallowed.\n");
        exit(EXIT_FAILURE);   
    }

    // retrieve and print info about gpu
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    printf("\nGPU INFO:\n"
           "\tDevice name: %s\n"
           "\tMemory: %g gb\n",
           prop.name, prop.totalGlobalMem/(1.E9));
    
    // // register signal handler to take care of interruption and termination signals
    // signal( SIGINT,  signal_handler );
    // signal( SIGTERM, signal_handler );
    
    // read input file

    printf("\n>>> Setting parameters\n");
    
    std::ifstream file(argv[1],ifstream::in);
    if (!file)
    {
        cerr << "ERROR: unable to open input file: " << argv[1] << endl;
        exit(2);
    }
    string buff;
    string label;

    if (file.is_open()) {
        string line;
        int i, j, k;
        while (getline(file, line)) {
            // using printf() in all tests for consistency
            if (line[0] == '#') continue;

            for (i = 0; i <= line.length(); i++)
            {
                if (line[i] == ' ') {
                    break;
                }    
            }

            for (j = i; j <= line.length(); j++)
            {
                if (line[j] != ' ') {
                    break;
                }
            }

            for (k = j; k <= line.length(); k++)
            {
                if (line[k] == ' ') {
                    break;
                }
            }
            label = line.substr(0, i);
            buff = line.substr(j, k-j);

            if (label == "task_title") {
                task_title = buff;
                cout << "setting task_title to:" << task_title << endl;
                
            }
            else if (label == "nmax") {
                sscanf(buff.c_str(), "%d", &nmax);
                cout << "Setting nmax to: " << nmax << endl;
                
            }
            else if (label == "vibmax") {
                sscanf(buff.c_str(), "%d", &vibmax);
                cout << "Setting vibmax to: " << vibmax << endl;
                
            }
            else if (label == "sys_vibmax") {
                sscanf(buff.c_str(), "%d", &sys_vibmax);
                cout << "Setting sys_vibmax to: " << sys_vibmax << endl;
                
            }
            else if (label == "hw") {
                sscanf(buff.c_str(), "%lf", &hw);
                cout << "Setting vibration energy to: " << hw << endl;
                
            }
            else if (label == "calc_pl") {
                if (buff == ".true."){
                    calc_pl = true;
                    cout << "Will calculate all spectra " << endl;
                } else if(buff == ".false."){
                    calc_pl = false;
                    cout << "Will calculate only absorption " << endl;
                }
                
            }
            else if (label == "lorentzian") {
                if (buff == ".true."){
                    lorentzian = true;
                    cout << "Lineshape set to Lorentzian " << endl;
                } else if(buff == ".false."){
                    lorentzian = false;
                    cout << "Lineshape set to Gaussian " << endl;
                }
                
            }
            else if (label == "no_frenkel") {
                if (buff == ".true."){
                    no_frenkel = true;
                    cout << "no frenkel coupling will account " << endl;
                } else if(buff == ".false."){
                    no_frenkel = false;
                    cout << "frenkel coupling will account " << endl;
                }
                
            }
            else if (label == "periodic") {
                if (buff == ".true.") {
                    periodic = true;
                    cout << "periodic condition is on " << endl;
                }
                else if (buff == ".false.") {
                    periodic = false;
                    cout << "periodic condition is off " << endl;
                }
                
            }
            else if (label == "nearest_neighbor") {
                if (buff == ".true.") {
                    nearest_neighbor = true;
                    cout << "calc coupling from nearest neighbor " << endl;
                }
                else if (buff == ".false.") {
                    nearest_neighbor = false;
                    cout << "calc coupling from long range " << endl;
                }
                
            }
            else if (label == "lambda_n") {
                sscanf(buff.c_str(), "%lf", &lambda_n);
                cout << "Setting lambda_n to: " << lambda_n << endl;
                
            }
            else if (label == "lambda_z1") {
                sscanf(buff.c_str(), "%lf", &lambda_z1);
                cout << "Setting lambda_z1 to: " << lambda_z1 << endl;
                
            }
            else if (label == "lambda_z2") {
                sscanf(buff.c_str(), "%lf", &lambda_z2);
                cout << "Setting lambda_z2 to: " << lambda_z2 << endl;
                
            }
            else if (label == "agg_angle") {
                sscanf(buff.c_str(), "%lf", &agg_angle);
                cout << "Setting agg_angle to: " << agg_angle << endl;
                
            }
            else if (label == "dielectric") {
                sscanf(buff.c_str(), "%lf", &dielectric);
                cout << "Setting dielectric to: " << dielectric << endl;
                
            }
            else if (label == "abs_lw") {
                sscanf(buff.c_str(), "%lf", &abs_lw);
                cout << "Setting the abs linewidth to (cm-1): " << abs_lw << endl;
                
            }
            else if (label == "nz") {
                sscanf(buff.c_str(), "%lf", &nz);
                cout << "Setting zwitter energy to (cm-1): " << nz << endl;
                
            }
            else if (label == "tz") {
                sscanf(buff.c_str(), "%lf", &tz);
                cout << "Setting intra charge transfer to (cm-1): " << tz << endl;
                
            }
            else if (label == "spec_step") {
                sscanf(buff.c_str(), "%d", &spec_step);
                cout << "Setting spec_step to (cm-1): " << spec_step << endl;
                
            }
            else if (label == "spec_start_ab") {
                sscanf(buff.c_str(), "%lf", &spec_start_ab);
                cout << "Setting spec_start_ab to: " << spec_start_ab << endl;
                
            }
            else if (label == "spec_end_ab") {
                sscanf(buff.c_str(), "%lf", &spec_end_ab);
                cout << "Setting spec_end_ab to: " << spec_end_ab << endl;
                
            }
            else if (label == "spec_start_pl") {
                sscanf(buff.c_str(), "%lf", &spec_start_pl);
                cout << "Setting spec_start_pl to: " << spec_start_pl << endl;
                
            }
            else if (label == "spec_end_pl") {
                sscanf(buff.c_str(), "%lf", &spec_end_pl);
                cout << "Setting spec_end_pl to: " << spec_end_pl << endl;
                
            }
            else if (label == "xyz_file1") {
                xyz_file[0] = buff;
                cout << "Will read the xyz file: " << xyz_file[0] << endl;
                
            }
            else if (label == "xyz_file2") {
                xyz_file[1] = buff;
                cout << "Will read the xyz file: " << xyz_file[1] << endl;
                
            }
            // else if (label == "xyz_file3") {
            //     xyz_file[2] = buff;
            //     cout << "Will read the xyz file: " << xyz_file[2] << endl;
                
            // }
            else
                cout << "invalid label at line, " << label << buff << endl;
                //exit(2);
        }
        file.close();
    }
    
    
    // determine the number of blocks to launch on the gpu 
    // each thread parts of the matrix for diagonalization
    
    //set lambda variables
    es_lambda_z1[es_n - 1] = lambda_n;
    es_lambda_z1[es_z1 - 1] = lambda_z1;
    es_lambda_z1[es_z2 - 1] = lambda_n;
    es_lambda_z2[es_n - 1] = lambda_n;
    es_lambda_z2[es_z1 - 1] = lambda_n;
    es_lambda_z2[es_z2 - 1] = lambda_z2;

    if (vibmax == 0) {
        memset(es_lambda_z1, 0.0, sizeof(es_lambda_z1));
        memset(es_lambda_z2, 0.0, sizeof(es_lambda_z2));
    }

    int status = mkdir(task_title.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    chdir(task_title.c_str());
    
    //write the geometry file
    double angle = 55.0;
        angle =  angle/180.00*pi;
    double d = 4.300;
    double armlength = 5.9950;
    
    //molecule 1
    ofstream file2("mole1.dat");
    file2 << "8 1 0.0 0.0 0.0" << endl;
    file2 << "6 2 " << armlength*cos(-agg_angle/2)*sin(angle) <<" "<<
            armlength*sin(-agg_angle/2)*sin(angle) <<" "<< -armlength*cos(angle) << endl;
    file2 << "8 3 " << 2.0*armlength*cos(-agg_angle/2)*sin(angle) <<" "<<
            2.0*armlength*sin(-agg_angle/2)*sin(angle) << " 0.0" << endl;
    file2.close();
    
    //molecule 2
    ofstream file3("mole2.dat");
    file3 << "8 1 0.0 0.0 " << d << endl;
    file3 << "6 2 " << armlength*cos(agg_angle/2)*sin(angle) << " " <<
            armlength*sin(agg_angle/2)*sin(angle) <<" "<< d+armlength*cos(angle) << endl;
    file3 << "8 3 " << 2.0*armlength*cos(agg_angle/2)*sin(angle) <<" "<<
            2.0*armlength*sin(agg_angle/2)*sin(angle) <<" "<< d << endl;
    file3.close();

    int lattice_kount = 0; 
    for (int n1 = 1; n1 <= nmax; ++n1) {
        for (int anum1 = 1; anum1 <= anum; ++anum1) {
            for (int index = 1; index <= 3; ++index) {        //x,y,z
                lattice_kount += 1;
            }
        }
    }
    mol1pos = new double[lattice_kount];
    
    //build geometry
    int tmp, tmp1; 
    for (int n1 = 1; n1 <= nmax; ++n1) {
        string line;
        int anum2;

        printf("%s%s\n","Reading the xyz file: ", xyz_file[n1-1].c_str());
        ifstream file4(xyz_file[n1-1]);
        while (getline(file4, line)) {
            sscanf(line.c_str(), "%d %d", &tmp,&anum2);
            int i = 9*n1+3*anum2-12; 
            sscanf(line.c_str(), "%d %d %lf %lf %lf", &tmp,&tmp1, 
                &mol1pos[i],&mol1pos[i+1],&mol1pos[i+2]);
        }
    }
    // for (int n1 = 0; n1 < lattice_kount; n1++) {
    //     cout<< mol1pos[n1]<<endl;
    // }
    
    //monomer state index 
    for (int run = 1; run <= 2; run++) {
        mon_kount = 0; 
        for (int es_state = 1; es_state <= es_zmax; ++es_state) {
            for (int vib1 = 0; vib1 <= vibmax; ++vib1) {
                for (int vib2 = 0; vib2 <= vibmax; ++vib2) {
                    mon_kount += 1;
                    
                        if (run == 2){
                            mon_state[mon_kount - 1].es_state = es_state;
                            mon_state[mon_kount - 1].vib1 = vib1;
                            mon_state[mon_kount - 1].vib2 = vib2;
                        }
                }
            }
        }

        if (run == 1) {
            mon_state = new basis[mon_kount];
        }
    }
    cout << "monomer states are: " << mon_kount << endl;

    //system state index
    int *molecule_state = new int[nmax];

    for (int run = 1; run <= 2; run++) {
        sys_kount = 0;
        //cout << "system states are: " << endl;
        for (int sysnx = 0; sysnx < pow(mon_kount, nmax); ++sysnx) {
            for (int n = 0; n < nmax; ++n) {
                molecule_state[n] = get_mon_state(sysnx+1, n+1);
                //cout << "system states are: " << molecule_state[n] << endl;
            }
            
            if (get_numvib(molecule_state) > sys_vibmax) continue;
            sys_kount += 1;
            //cout << "system states" << get_numvib(molecule_state) << endl;
                if (run == 2) {
                     for (int n = 0; n < nmax; ++n) {
                        sys_state[sys_kount-1][n] = mon_state[molecule_state[n]-1];
                     }
                 }
        }
        if (run == 1) {
            sys_state = new basis*[sys_kount];
            for (int i = 0; i < sys_kount; ++i) 
                sys_state[i] = new basis[nmax];
        }
    }

    cout << "system states are: " << sys_kount << endl;
    // for (int sysnx = 0; sysnx < sys_kount; ++sysnx) {
        // for (int n = 0; n < nmax; ++n) {
            // cout << sys_state[sys_kount][n].es_state << endl;
            // cout << sys_state[sys_kount][n].vib1 << endl;
            // cout << sys_state[sys_kount][n].vib2 << endl;
        // }
    // }

    // coulomb coupling calculation
    coulomb_coupling = new double[sys_kount];
    for (int sysnx = 0; sysnx < sys_kount; ++sysnx) {
        coulomb_coupling[sysnx] = 0.0; 
        //cout << coulomb_coupling[sysnx] << endl;
    }
    
    const double eo         = 8.854187817*pow(10,-12); //(f/m)
    const double plancks    = 6.62606957*pow(10,-34);  //kg m**2 s**-2
    const double csol       = 2.99792458*pow(10,8);    //m s**-1
    const double aucharge_c = 1.602176565*pow(10,-19);  //c per au
    //const double au_debye   = 2.54175;       //au per debye
    double coeff = pow(aucharge_c,2)*(pow(10,10))/(4.0*pi*100.0*eo*plancks*csol)/dielectric; 
    
    if( nmax >=2 ){
        for (int sysnx = 1; sysnx <= sys_kount; ++sysnx){
            for (int n1 = 1; n1 < nmax; ++n1) {
                int state1 = sys_state[sysnx-1][n1-1].es_state; 

            for (int n2 = n1+1; n2 <= nmax; ++n2) {
                int state2 = sys_state[sysnx-1][n2-1].es_state; 
                    
                for (int da1 = 1; da1 <= anum; ++da1) {
                for (int da2 = 1; da2 <= anum; ++da2) {
                    // cout << get_distance(n1,da1,n2,da2) << endl;
                    
                    coulomb_coupling[sysnx-1]+= get_charge(state1, da1)
                            *get_charge(state2, da2)
                            /get_distance(n1,da1,n2,da2); 
                    // cout << coulomb_coupling[sysnx-1] << endl;

                }
                }
                
            }
            }
            coulomb_coupling[sysnx-1] = coulomb_coupling[sysnx-1] * coeff; 
            // cout << coulomb_coupling[sysnx-1] << endl;
        }
    }

    if(no_frenkel){
	    for (int sysnx = 0; sysnx < sys_kount; ++sysnx) {
	        coulomb_coupling[sysnx] = 0.0; 
	        //cout << coulomb_coupling[sysnx] << endl;
	    }	
    }
    
    // for (int sysnx = 0; sysnx < sys_kount; ++sysnx) {
        // cout << coulomb_coupling[sysnx] << endl;
    // }

    // transition dipole moment
    ux = new double[sys_kount];
    uy = new double[sys_kount];
    uz = new double[sys_kount];
    for (int sysnx = 0; sysnx < sys_kount; ++sysnx) {
        ux[sysnx] = 0.0; 
        uy[sysnx] = 0.0;
        uz[sysnx] = 0.0; 
    }
    

    for (int sysnx = 1; sysnx <= sys_kount; ++sysnx){
        for (int n1 = 1; n1 < nmax; ++n1) {
            int state1 = sys_state[sysnx-1][n1-1].es_state; 
            for (int da1 = 1; da1 <= anum; ++da1) {
                int i = 9*n1 + 3*da1 - 12; 
                ux[sysnx-1] += get_charge(state1, da1)* mol1pos[i];
                uy[sysnx-1] += get_charge(state1, da1)* mol1pos[i+1];
                uz[sysnx-1] += get_charge(state1, da1)* mol1pos[i+2];
            }
        }
    }


    //create hamiltonian 
    sys_h = new double[sys_kount*sys_kount];
    
    
    //initial matrix
    for (int state1 = 0; state1 < sys_kount; ++state1) {
        //d_w[state1] = 0.0; 
        for (int state2 = 0; state2 < sys_kount; ++state2) {
            sys_h[state1*sys_kount + state2] = 0.0;
        }
    }

    //on diagonal
    for (int state1 = 0; state1 < sys_kount; ++state1) {
        for (int n1 = 0; n1 < nmax; ++n1) {
            int state_a = sys_state[state1][n1].es_state;
            int vib_a1 = sys_state[state1][n1].vib1;
            int vib_a2 = sys_state[state1][n1].vib2;
            //cout << state_a << vib_a1 << vib_a2 << endl;
            //the electronic energy
            if (state_a == es_z1) {
                sys_h[state1*sys_kount + state1] += nz;
            }
            else if (state_a == es_z2) {
                sys_h[state1*sys_kount + state1] += nz;
            }
            //the vibration energy
            sys_h[state1*sys_kount + state1] += (vib_a1 + vib_a2)*hw; 
        }
        //Coulombic coupling for dimer 
        sys_h[state1*sys_kount + state1] += coulomb_coupling[state1]; 
        // cout << "count " << state1*sys_kount + state1<<sys_h[state1*sys_kount + state1] << endl;
    }

    //off diagonal
    for (int state1 = 0; state1 < sys_kount; ++state1) {
        for (int state2 = 0; state2 < sys_kount; ++state2) {
            if (state1 == state2) continue;
            //intramolecular CT
            int diff = 0;
            int diffn = 0;
            for (int n1 = 0; n1 < nmax; ++n1) {
                int state_a = sys_state[state1][n1].es_state;
                int vib_a1 = sys_state[state1][n1].vib1;
                int vib_a2 = sys_state[state1][n1].vib2;
                int state_b = sys_state[state2][n1].es_state;
                int vib_b1 = sys_state[state2][n1].vib1;
                int vib_b2 = sys_state[state2][n1].vib2;

                if (vib_a1 != vib_b1 || state_a != state_b || vib_a2 != vib_b2) {
                    diffn = n1;
                    diff += 1;
                }
                if (diff > 1) continue;
            }
            //at max, only one molecule can have a different configuration
            if (diff == 1) {
                int state_a = sys_state[state1][diffn].es_state;
                int vib_a1 = sys_state[state1][diffn].vib1;
                int vib_a2 = sys_state[state1][diffn].vib2;
                int state_b = sys_state[state2][diffn].es_state;
                int vib_b1 = sys_state[state2][diffn].vib1;
                int vib_b2 = sys_state[state2][diffn].vib2;

                if (state_a == es_n && state_b == es_z1) {
                    sys_h[state1*sys_kount + state2] = tz;
                }
                else if (state_a == es_n && state_b == es_z2) {
                    sys_h[state1*sys_kount + state2] = tz;
                }
                else if (state_a == es_z1 && state_b == es_n) {
                    sys_h[state1*sys_kount + state2] = tz;
                }
                else if (state_a == es_z2 && state_b == es_n) {
                    sys_h[state1*sys_kount + state2] = tz;
                }
                else {
                    sys_h[state1*sys_kount + state2] = 0.0;
                }

                //put vibration into
                double volapfact = volap(es_lambda_z1[state_a], vib_a1, 
                    es_lambda_z1[state_b], vib_b1)
                    *volap(es_lambda_z2[state_a], vib_a2, 
                        es_lambda_z2[state_b], vib_b2); 
                sys_h[state1*sys_kount + state2] = sys_h[state1*sys_kount + state2] * volapfact;
            }

        }
    }
    //print out the hamiltonian
    if (sys_kount < 100) {
        FILE* stream = fopen((task_title + "_H.dat").c_str(),"w");
        MKL_INT i,j; 
        
        //fprintf(stream, "Printing Matrix : \n");
        fprintf(stream, "\n %s\n", "Hamiltonian" );
        
        for( i = 0; i < sys_kount; ++i ) {
            for( j = 0; j < sys_kount; ++j ) fprintf(stream, " %6.2lf", sys_h[i*sys_kount+j]);
            fprintf(stream, "\n" ); 
        }
        //printf( "The algorithm failed to compute eigenvalues.\n" );
        fclose(stream);
        //if (outfile.is_open()) {
            //outfile << "Printing Matrix : \n";
            // for (int state1 = 0; state1 < sys_kount; ++state1) {
                // for (int state2 = 0; state2 < sys_kount; ++state2) {
                    // outfile << *(*(sys_h + state1) + state2) << " ";
                // }
                // outfile << endl;
            // }
            // outfile.close();
    }

    // magma variables for magma matrix print 
    magma_queue_t       queue;
    cudaError_t         Cuerr;
    int                 Merr;
    
    // magma variables for dsyevd
    double         *d_sys_h;                                                       // the hamiltonian on the GPU
    double         aux_work[1];                                                    // To get optimal size of lwork
    magma_int_t         aux_iwork[1], info;                                             // To get optimal liwork, and return info
    magma_int_t         lwork, liwork;                                                  // Leading dim of kappa, sizes of work arrays
    magma_int_t         *iwork;                                                         // Work array
    double         *work;                                                          // Work array
    double         *w   ;                                                          // Eigenvalues
    double         *wA  ;                                                          // Work array
    
    //double         *d_w;                                                           // Eigenvalues on the GPU

    // Initialize magma math library and queue
    magma_init(); magma_queue_create( 0, &queue ); 
    
#define CHK_ERR     if (Cuerr != cudaSuccess ) { printf(">>> ERROR on CUDA: %s.\n", cudaGetErrorString(Cuerr)); exit(EXIT_FAILURE);}
#define MALLOC_ERR  { printf(">>> ERROR on CPU: out of memory.\n"); exit(EXIT_FAILURE);}
#define CHK_MERR    if (Merr != MAGMA_SUCCESS ) { printf(">>> ERROR on MAGMA: %s.\n", magma_strerror(Merr)); exit(EXIT_FAILURE);}
    int SSYEVD_ALLOC_FLAG = 1;     // flag whether to allocate ssyevr arrays -- it is turned off after they are allocated
    
    magma_int_t sys_kount2 = (magma_int_t) sys_kount*sys_kount; 
    
    //allocate memory in GPU
    Cuerr = cudaMalloc (&d_sys_h,sys_kount2*sizeof(double)); CHK_ERR;
    


    //begin the diagonalization
    cudaMemcpy( d_sys_h, sys_h, sys_kount2* sizeof(double), cudaMemcpyHostToDevice );

    // if the first time, query for optimal workspace dimensions
    if ( SSYEVD_ALLOC_FLAG )
    {   
        magma_dsyevd_gpu( MagmaVec, MagmaUpper, (magma_int_t) sys_kount, NULL, (magma_int_t) sys_kount, 
            NULL, NULL, (magma_int_t) sys_kount, aux_work, -1, aux_iwork, -1, &info );
        
        lwork  = (magma_int_t) MAGMA_D_REAL( aux_work[0] );
        liwork  = aux_iwork[0];

        // allocate work arrays, eigenvalues and other stuff
        
        Merr = magma_imalloc_cpu   ( &iwork, liwork ); CHK_MERR; 
        Merr = magma_dmalloc_pinned( &wA , sys_kount2 ) ; CHK_MERR;
        Merr = magma_dmalloc_cpu   ( &w,     sys_kount ); CHK_MERR; 
        Merr = magma_dmalloc_pinned( &work , lwork  ); CHK_MERR;

        SSYEVD_ALLOC_FLAG = 0;      // is allocated here, so we won't need to do it again
        //cout<< "Hamiltonian" <<endl;

        // get info about space needed for diagonalization
        size_t freem, total;
        cudaMemGetInfo( &freem, &total );
        printf("\n>>> cudaMemGetInfo returned\n"
               "\tfree:  %g gb\n"
               "\ttotal: %g gb\n", (double) freem/(1E9), (double) total/(1E9));
        printf(">>> %g gb needed by diagonalization routine.\n", (double) (lwork * (double) sizeof(double)/(1E9)));
    }

    magma_dsyevd_gpu( MagmaVec, MagmaUpper, (magma_int_t) sys_kount, d_sys_h, (magma_int_t) sys_kount,
        w, wA, (magma_int_t) sys_kount, work, lwork, iwork, liwork, &info );

    if ( info != 0 ){ printf("ERROR: magma_dsyevd_gpu returned info %lld.\n", info ); exit(EXIT_FAILURE);}
                
    // copy eigenvalues to device memory
    //cudaMemcpy( d_w    , w    , sys_kount*sizeof(double), cudaMemcpyDeviceToHost );
    cudaMemcpy( sys_h, d_sys_h, sys_kount2* sizeof(double), cudaMemcpyDeviceToHost );
    
    //test
    // magma_dprint_gpu((magma_int_t) sys_kount,(magma_int_t) sys_kount,
    // d_sys_h,(magma_int_t) sys_kount,queue); 	
    
    //magma_dprint( (magma_int_t) sys_kount, (magma_int_t) sys_kount, sys_h, (magma_int_t) sys_kount); 

    //transpose
    double temp; 
    int i,j;
    for( i = 0; i < sys_kount; ++i ) {
        for( j = i; j < sys_kount; ++j ) {
            temp = sys_h[i*sys_kount+j];
            sys_h[i*sys_kount+j] = sys_h[j*sys_kount+i]; 
            sys_h[j*sys_kount+i] = temp ; 
        }
    }

    //print eigenvalue & eigenvectors
    if (sys_kount < 100) {
        FILE* stream = fopen((task_title + "_evec.dat").c_str(),"w");
        MKL_INT i,j; 
        
        //fprintf(stream, "Printing Matrix : \n");
        fprintf(stream, "\n %s\n", "Eigenvalues" );
        for( j = 0; j < sys_kount; ++j ) fprintf(stream, " %6.2lf", w[j] );
        fprintf(stream, "\n" );
        
        fprintf(stream, "\n %s\n", "Eigenvectors (stored columnwise)" );
        
        for( i = 0; i < sys_kount; ++i ) {
            for( j = 0; j < sys_kount; ++j ) fprintf(stream, " %6.4lf", sys_h[i*sys_kount+j]);
            fprintf( stream, "\n" );
        }
    
        fclose(stream);
    }
    
    //absorption spectrum
    //absorption oscilator strength
    //double lineshape; 
    ab_osc_x = new double[sys_kount-1];
    ab_osc_y = new double[sys_kount-1];
    ab_osc_z = new double[sys_kount-1];
    ab_sys_eval = new double[sys_kount-1];

    for (int sysnx = 0; sysnx < sys_kount-1; ++sysnx) {
        ab_osc_x[sysnx] = 0.0; 
        ab_osc_y[sysnx] = 0.0;
        ab_osc_z[sysnx] = 0.0; 
    }
    
    int state; 
    
    for (state = 1; state < sys_kount; ++state){
        ab_sys_eval[state-1] = *(w + state) - *(w); 
        for(int hx = 0; hx < sys_kount; ++hx ){
            ab_osc_x[state-1] += ux[hx] * sys_h[hx*sys_kount+state] * sys_h[hx*sys_kount];
            ab_osc_y[state-1] += uy[hx] * sys_h[hx*sys_kount+state] * sys_h[hx*sys_kount];
            ab_osc_z[state-1] += uz[hx] * sys_h[hx*sys_kount+state] * sys_h[hx*sys_kount];
        }
        ab_osc_x[state-1] = pow(ab_osc_x[state-1],2);
        ab_osc_y[state-1] = pow(ab_osc_y[state-1],2);
        ab_osc_z[state-1] = pow(ab_osc_z[state-1],2);
    }
    
     //find the highest oscillator strength
    double temp_x = 0;
    double temp_y = 0;
    //double temp_z = 0;
    
    int pl_start_x, pl_start_y; 
    
    for (state = 0; state < sys_kount-1; state++){
        if (ab_osc_x[state]>temp_x){
            temp_x = ab_osc_x[state];
            pl_start_x = state;
        }
        if (ab_osc_y[state]>temp_y){
            temp_y = ab_osc_y[state];
            pl_start_y = state;
        }
    }

        
    pl_start = min(pl_start_x,pl_start_y); 
    printf( "lowest excited state is: %d \n" , pl_start);
    
    //absorption spectrum
    ab_x = new double[spec_step];
    ab_y = new double[spec_step];
    ab_z = new double[spec_step];
    for (int spec_point = 0; spec_point < spec_step; ++spec_point) {
        ab_x[spec_point] = 0.0; 
        ab_y[spec_point] = 0.0;
        ab_z[spec_point] = 0.0; 
    }
    
    for (int spec_point = 0; spec_point < spec_step; spec_point++){
        double energy = spec_start_ab + (spec_end_ab - spec_start_ab)/spec_step*(spec_point+1); 
        for(int state = 0; state < sys_kount-1; state++ ){
            double tran_e = ab_sys_eval[state];
            double lineshape; 
            if (lorentzian){
                lineshape = abs_lw/(pow((energy-tran_e),2)+pow(abs_lw,2))/pi;
            }
            else{
                lineshape = exp(-(pow((energy - tran_e)/abs_lw,2)));
            }
            
            if ( abs_freq_dep ){
                ab_x[spec_point] += lineshape * ab_osc_x[state] * tran_e/pow(10.0,4);
                ab_y[spec_point] += lineshape * ab_osc_y[state] * tran_e/pow(10.0,4);
                ab_z[spec_point] += lineshape * ab_osc_z[state] * tran_e/pow(10.0,4);
            }
            else{
                ab_x[spec_point] += lineshape * ab_osc_x[state];
                ab_y[spec_point] += lineshape * ab_osc_y[state];
                ab_z[spec_point] += lineshape * ab_osc_z[state];
            }
        }
    }
    
    //print absorption spectrum
    FILE* stream = fopen((task_title + "_ab.dat").c_str(),"w");
    
        // fprintf(stream, "Printing Matrix : \n");
        fprintf(stream, "%s\n", "Energy A(g(w))" );
        fprintf(stream, "%s\n", "Energy System" );
        fprintf(stream, "%s\n\n", "cm +(-1) a.u." );
        
        for(int spec_point = 0; spec_point < spec_step; ++spec_point ){
            double energy = spec_start_ab + (spec_end_ab - spec_start_ab)/spec_step*(spec_point+1);
            fprintf(stream, " %lf %lf %lf %lf %lf\n", energy, ab_x[spec_point]+ab_y[spec_point]+ab_z[spec_point],
                    ab_x[spec_point], ab_y[spec_point], ab_z[spec_point]);
        }

    fclose(stream);
    
    // pl spectrum
    // pl oscilator strength
    pl_osc_x = new double[pl_start-1];
    pl_osc_y = new double[pl_start-1];
    pl_osc_z = new double[pl_start-1];
    pl_sys_eval = new double[pl_start-1];
    for (int sysnx = 0; sysnx < pl_start-1; ++sysnx) {
        pl_osc_x[sysnx] = 0.0; 
        pl_osc_y[sysnx] = 0.0;
        pl_osc_z[sysnx] = 0.0; 
    }
    
    for (int state = 1; state < pl_start; ++state){
        pl_sys_eval[state-1] = *(w + pl_start) - *(w + state - 1); 
        for(int hx = 0; hx < sys_kount; hx++ ){
            pl_osc_x[state-1] += ux[hx] * sys_h[hx*sys_kount+state-1] * sys_h[hx*sys_kount+pl_start];
            pl_osc_y[state-1] += uy[hx] * sys_h[hx*sys_kount+state-1] * sys_h[hx*sys_kount+pl_start];
            pl_osc_z[state-1] += uz[hx] * sys_h[hx*sys_kount+state-1] * sys_h[hx*sys_kount+pl_start];
        }
        pl_osc_x[state-1] = pl_osc_x[state-1] * pl_osc_x[state-1];
        pl_osc_y[state-1] = pl_osc_y[state-1] * pl_osc_y[state-1];
        pl_osc_z[state-1] = pl_osc_z[state-1] * pl_osc_z[state-1];
    }
    
    //pl spectrum
    pl_x = new double[spec_step];
    pl_y = new double[spec_step];
    pl_z = new double[spec_step];
    for (int spec_point = 0; spec_point < spec_step; ++spec_point) {
        pl_x[spec_point] = 0.0; 
        pl_y[spec_point] = 0.0;
        pl_z[spec_point] = 0.0; 
    }
    
    for (int spec_point = 0; spec_point < spec_step; ++spec_point){
        double energy = spec_start_pl + (spec_end_pl - spec_start_pl)/spec_step*(spec_point+1);
        for(int state = 0; state < pl_start-1; ++state ){
            double tran_e = pl_sys_eval[state]; 
            double lineshape; 
            if (lorentzian){
                lineshape = abs_lw/(pow((energy-tran_e),2)+pow(abs_lw,2))/pi; 
            }
            else{
                lineshape = exp(-pow((energy - tran_e)/abs_lw,2)); 
            }
            
            pl_x[spec_point] += lineshape * pl_osc_x[state];
            pl_y[spec_point] += lineshape * pl_osc_y[state];
            pl_z[spec_point] += lineshape * pl_osc_z[state];

            if ( abs_freq_dep ){
                pl_x[spec_point] += lineshape * pl_osc_x[state] * pow(tran_e,3)/pow(10.0,12);
                pl_y[spec_point] += lineshape * pl_osc_y[state] * pow(tran_e,3)/pow(10.0,12);
                pl_z[spec_point] += lineshape * pl_osc_z[state] * pow(tran_e,3)/pow(10.0,12);
            }
            else{
                pl_x[spec_point] += lineshape * pl_osc_x[state];
                pl_y[spec_point] += lineshape * pl_osc_y[state];
                pl_z[spec_point] += lineshape * pl_osc_z[state];
            }
        }
    }
    
    //print absorption spectrum
    FILE* stream1 = fopen((task_title + "_pl.dat").c_str(),"w");
    
        // fprintf(stream, "Printing Matrix : \n");
        fprintf(stream1, "%s\n", "Energy F(g(w))" );
        fprintf(stream1, "%s\n", "Energy System" );
        fprintf(stream1, "%s\n\n", "cm +(-1) a.u." );
        
        for(int spec_point = 0; spec_point < spec_step; ++spec_point ){
            double energy = spec_start_pl + (spec_end_pl - spec_start_pl)/spec_step*(spec_point+1);
            fprintf(stream1, " %lf %lf %lf %lf %lf\n", energy, pl_x[spec_point]+pl_y[spec_point]+pl_z[spec_point],
                    pl_x[spec_point], pl_y[spec_point], pl_z[spec_point]);
        }

    fclose(stream1);
    
    //paremeters out
    ofstream file1(task_title + "_para.csv");
    file1 << "parameter, value" << endl;
    file1 << "@@@@@@@@@@@@@@@@" << endl;
    file1 << "task title, " << task_title << endl;
    file1 << "es_zmax, " << es_zmax << endl;
    file1 << "nmax, " << nmax << endl;
    file1 << "vibrational energy (cm-1), " << hw << endl;
    file1 << "Ion pair energy (cm-1), " << nz << endl;
    file1 << "Intramolecule charge transfer (cm-1), " << tz << endl;
    file1 << "vibmax, " << vibmax << endl;
    file1 << "sys_vibmax, " << sys_vibmax << endl;
    file1 << "lambda_n, " << lambda_n << endl;
    file1 << "lambda_z1, " << lambda_z1 << endl;
    file1 << "lambda_z2, " << lambda_z2 << endl;
    file1 << "monomer kount, " << mon_kount << endl;
    file1 << "system kount, " << sys_kount << endl;
    file1 << "abs linewidth (cm-1), " << abs_lw  << endl;
    file1 << "dielectric, " << dielectric  << endl;
    file1 << "no_frenkel, " << no_frenkel  << endl;
    file1 << "periodic, " << periodic  << endl;    
    file1.close();
    

    end = time(NULL);
    printf("\n>>> Done with the calculation in %f seconds.\n", difftime(end,start));

    
    //free memory 
    // delete[] eigenvector;

    // for (int i = 0; i < sys_kount; i++)
    //     delete[] sys_h[i];
    delete[] sys_h;

    
    delete[] mon_state; 
    for (int i = 0; i < sys_kount; i++)
        delete[] sys_state[i];
    delete[] sys_state;


    delete[] ab_osc_x;
    delete[] ab_osc_y;
    delete[] ab_osc_z;      
    delete[] ab_x;    
    delete[] ab_y;    
    delete[] ab_z; 

    delete[] ux;    
    delete[] uy;    
    delete[] uz;    

    delete[] pl_osc_x;
    delete[] pl_osc_y;
    delete[] pl_osc_z;      
    delete[] pl_x;    
    delete[] pl_y;    
    delete[] pl_z; 

    delete[] coulomb_coupling;
    delete[] mol1pos;

    

    // free memory on the CPU and GPU and finalize magma library
    
    cudaFree(d_sys_h);
    
    //cudaFree(d_w);
    if ( SSYEVD_ALLOC_FLAG == 0 )
    {
        free(w);
        free(iwork);
        magma_free_pinned( work );
        magma_free_pinned( wA );
    }
    
    exit(0);
    
    // auto stop = high_resolution_clock::now(); 
    // auto duration = duration_cast<microseconds>(stop - start);
    // cout << duration.count() << endl; 

    // t2=clock();
    // float diff ((float)t2-(float)t1);
    // cout<<diff<<endl;

    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize (stop) ;
    // cudaEventElapsedTime(&elapsed, start, stop) ;
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);


    // final call to finalize magma math library
    magma_finalize();

    return 0;

}

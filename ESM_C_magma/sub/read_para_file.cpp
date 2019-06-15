#include "esm_c_magma.h" 


// Global variable to catch interrupt and terminate signals
volatile sig_atomic_t interrupted=false;

void read_para_file(filename){


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
}
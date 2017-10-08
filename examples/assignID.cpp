#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>


bool sortbydes(const std::pair<std::string,int> &a, 
              const std::pair<std::string,int> &b) {
    return (a.second < b.second);
}

int main(int argc, char** argv) {
    // argv[1] test file name
    // argv[2] benchmark file name
    int topk = std::stoi(argv[3]);
    //
	std::ifstream file1(argv[1]);
	std::string line1;
	std::vector<std::set<std::string>> test_result;
	while(std::getline(file1, line1)) {
		std::vector<std::string> base_row;
    	std::stringstream linestream(line1);
    	std::string s;
    	std::getline(linestream, s); 

		std::string delimiter = " ";
		size_t pos = 0;
		std::string token;
		std::string tmp;
        std::set<std::string> row_set;
        int cnt = 0;
		while ((pos = s.find(delimiter)) != std::string::npos) {
    		token = s.substr(0, pos);
            if (cnt != 0) {
			    tmp = token;
                row_set.insert(tmp);
            }
    		s.erase(0, pos + delimiter.length());
            cnt ++;
		}
		tmp = s;
        row_set.insert(tmp);
		test_result.push_back(row_set);
		row_set.clear();
	}
    file1.close();

	std::ifstream file2(argv[2]);
	std::string line2;
	std::vector<std::set<std::string>> bench_result;
	while(std::getline(file2, line2)) {
		std::vector<std::string> bench_row;
    	std::stringstream linestream(line2);
    	std::string s;
    	std::getline(linestream, s); 

		std::string delimiter = " ";
		size_t pos = 0;
		std::string token;
		std::string tmp;
        std::set<std::string> row_set;
        int cnt = 0;
		while ((pos = s.find(delimiter)) != std::string::npos) {
    		token = s.substr(0, pos);
            if (cnt != 0) {
			    tmp = token;
                row_set.insert(tmp);
            }
    		s.erase(0, pos + delimiter.length());
            cnt ++;
		}
		tmp = s;
        row_set.insert(tmp);
		bench_result.push_back(row_set);
		row_set.clear();
	}
    file2.close();

    int sum = 0;
    for (int i = 0; i < 100; i++ ) {
		std::vector<std::string>::iterator it;
		std::vector<std::string> v(2*topk);
  		it=std::set_intersection ((test_result[i]).begin(), (test_result[i]).end(), 
			(bench_result[i]).begin(), (bench_result[i]).end(), v.begin());
		v.resize(it-v.begin());
		sum += v.size();
	}
    double precision = 1.0 * sum / (topk*100);
    std::cout << precision;


	return 0;
}

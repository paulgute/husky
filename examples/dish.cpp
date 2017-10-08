// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "io/hdfs_manager.hpp"
#include "io/input/line_inputformat.hpp"
#include "io/input/binary_inputformat.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"

#include "Eigen/LU"
#include "Eigen/SparseCore"
#include "Eigen/SVD"

// serialization for eigen
namespace husky {

BinStream & operator >> (BinStream & in, Eigen::MatrixXd & mat) {
    size_t len_row, len_col;
    in >> len_row >> len_col;
    mat = Eigen::MatrixXd(len_row, len_col);
    double val;
    for (int i = 0; i < len_row; i++) {
        for (int j = 0; j < len_col; j++) {
            in >> i >> j >> val;
            mat.coeffRef(i, j) = val;
        }
    }
    return in;
}

BinStream & operator << (BinStream & out, const Eigen::MatrixXd & mat) {
    out << size_t(mat.rows());
    out << size_t(mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            out << i << j << double(mat(i, j));
        }
    }
    return out;
}

BinStream & operator >> (BinStream & in, Eigen::VectorXd & vec) {
    size_t len;
    in >> len;
    vec.resize(len);
    double val;
    for (size_t i = 0; i < len; i++) {
        in >> val;
        vec.coeffRef(i) = val;
    }
    return in;
}

BinStream & operator << (BinStream & out, const Eigen::VectorXd & vec) {
    size_t len = vec.size();
    out << len;
    for (size_t i = 0; i < len; i++) {
        out << double(vec(i));
    }
    return out;
}
// End serialization for Eigen

class VectorObject {
public:
    typedef std::string KeyT;
    KeyT name;
    Eigen::VectorXd vec;

    const KeyT & id() const { return name; }

    void key_init(KeyT name) {
        this->name = name;
    }

    friend BinStream & operator >> (BinStream & stream, VectorObject & r) {
        stream >> r.name >> r.vec;
        return stream;
    }

    friend BinStream & operator << (BinStream & stream, VectorObject & r) {
        stream << r.name << r.vec;
        return stream;
    }
};

class MatrixObject {
public:
    typedef std::string KeyT;
    KeyT name;
    Eigen::MatrixXd mat;

    const KeyT & id() const { return name; }

    void key_init(KeyT name) {
        this->name = name;
    }

    friend BinStream & operator >> (BinStream & stream, MatrixObject & r) {
        stream >> r.name >> r.mat;
        return stream;
    }

    friend BinStream & operator << (BinStream & stream, MatrixObject & r) {
        stream << r.name << r.mat;
        return stream;
    }
};

class ImageObject {
public:
    typedef std::string KeyT;
    KeyT name;
    Eigen::MatrixXd mat;
    Eigen::MatrixXd code;

    const KeyT & id() const { return name; }

    void key_init(KeyT name) {
        this->name = name;
    }

    friend BinStream & operator >> (BinStream & stream, ImageObject & r) {
        stream >> r.name >> r.mat >> r.code;
        return stream;
    }

    friend BinStream & operator << (BinStream & stream, ImageObject & r) {
        stream << r.name << r.mat << r.code;
        return stream;
    }
};

class NodeObject { // for each node, we have C the dictionary, A the lagrangian multiplier, p the penalty parameter, B the binary matrix, X the data
public:
    typedef std::string KeyT;
    KeyT name;
    Eigen::MatrixXd new_C;
    Eigen::MatrixXd old_C;
    Eigen::MatrixXd A;
    Eigen::MatrixXd new_B;
    Eigen::MatrixXd old_B;
    Eigen::MatrixXd X;
    double p;
    int err;
    std::vector<std::string> name_list;

    const KeyT & id() const { return name; }

    void key_init(KeyT name) {
        this->name = name;
    }

    friend BinStream & operator >> (BinStream & stream, NodeObject & r) {
        stream >> r.name >> r.new_C >> r.old_C >> r.A >> r.new_B >> r.old_B >> r.X >> r.p >> r.err >> r.name_list;
        return stream;
    }

    friend BinStream & operator << (BinStream & stream, NodeObject & r) {
        stream << r.name << r.new_C << r.old_C << r.A << r.new_B << r.old_B << r.X << r.p << r.err << r.name_list;
        return stream;
    }
};



Eigen::MatrixXd binary_convert(Eigen::MatrixXd input, int r, int n) {
    Eigen::MatrixXd output(r, n);
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < n; j++) {
            if(input(i, j) < 0 || input(i,j) == 0) {
                output(i, j) = 0;}
            else {
                output(i, j) = 1;}
        }
    }
    return output;
}

void DISH() {
    // get the data url
    std::string learn_url = Context::get_param("learn_url");
    // get the number of epoch 
    int max_epoch = std::stoi(Context::get_param("max_epoch"));
    // get the number of learn_data 
    int N = std::stoi(Context::get_param("N"));
    // get the number of ADMM iterations
    int K = std::stoi(Context::get_param("K"));
    // get the penalty parameter 
    double p = std::stoi(Context::get_param("p"));
    // get dimension of data points 
    int d = std::stoi(Context::get_param("d"));
    // get dimension of binary code 
    int r = std::stoi(Context::get_param("r"));

    // read data
    husky::io::BinaryInputFormat infmt_learn(learn_url);
    // indices to track the reading process
    int cur_col = 0;
    long n = 0;
    // use a list of vector objects to store the data
    auto & vector_data_list = ObjListStore::create_objlist<VectorObject>("vectors of data");
    // use a BinStream reference to read
    husky::load(infmt_learn, [&](husky::base::BinStream& file) {
        int size = file.size();
        std::vector<double> row_vector;
        while (file.size()) {
            float sz = husky::base::deser<float>(file);  
            if (cur_col != 0) {
                row_vector.push_back(sz);
            }
            cur_col ++;
            if (cur_col == d+1) {
                std::string key = "learn" + std::to_string(Context::get_global_tid()) + std::to_string(n);
                VectorObject X;
                X.key_init(key);
                Eigen::VectorXd x(d);
                for (int k = 0; k < d; k++) {
                    x(k) = row_vector[k];
                }
                X.vec = x;
                vector_data_list.add_object(X); 
                row_vector.clear();
                cur_col = 0;
                n++;
            }
        }
        LOG_I << "total " << n << " learn data";
    });
    globalize(vector_data_list);
    if (Context::get_global_tid() == 0)
        LOG_I << "reading learn fvecs and globalization for vector of data done";

    // calculate the mean of the data
    std::vector<unsigned long long> sum_vector(d);
    std::vector<unsigned long long> zero_vector(d);
    lib::Aggregator<std::vector<unsigned long long>> data_sum_agg(zero_vector, [&](std::vector<unsigned long long>& a, const std::vector<unsigned long long>& b){ 
        for (int i = 0; i < d; i++) {
            a[i] += b[i];
        }
    }, [&](std::vector<unsigned long long>& v){ v = std::move(std::vector<unsigned long long>(d));});
    list_execute(vector_data_list, [&](VectorObject& obj) {
        std::vector<unsigned long long> tmp(d);
        for (int i = 0; i < d; i++) {
            tmp[i] = (obj.vec)(i);
        }
        data_sum_agg.update(tmp);
    });
    lib::AggregatorFactory::sync();
    sum_vector = data_sum_agg.get_value();
    Eigen::VectorXd Mean_Vector(d);
    for (int i = 0; i < d; i++) {
            Mean_Vector(i) = sum_vector[i]/N;
    }
    if (Context::get_global_tid() == 0) {
    //    LOG_I << "Mean Vector is " << Mean_Vector;
        LOG_I << "Mean vector done";
    }

    // centralize the data
    list_execute(vector_data_list, [&](VectorObject& obj) {
        obj.vec = obj.vec - Mean_Vector;
    });

    // store the data into a local raw matrix
    std::vector<std::vector<double>> raw_matrix;
    std::vector<std::string> name_list;
    n = 0;
    list_execute(vector_data_list, [&](VectorObject& obj) {
        std::vector<double> v;
        for (int i = 0; i < d; i++) {
             float tmp = obj.vec(i);
             v.push_back(tmp);
        }
        raw_matrix.push_back(v);
        name_list.push_back(obj.name);
        n++;
    });
    ObjListStore::drop_objlist("vectors of data");

    // store the raw matrix into a MatrixXD 
   // LOG_I << "worker " <<  std::to_string(Context::get_global_tid()) << " has " << n << " rows";

       

    Eigen::MatrixXd mat(d, n);
    for (int i = 0; i< n; i++) {
        for(int j = 0; j < d; j++) {
            mat(j, i) = (raw_matrix[i])[j];
        }
    }

    // initialize parameters
    // initialize dictionary C
    Eigen::MatrixXd ZERO_d_r(d, r);
    lib::Aggregator<Eigen::MatrixXd> C_initialize_agg(ZERO_d_r, [](Eigen::MatrixXd& a, const Eigen::MatrixXd& b){ a = b; });
    Eigen::MatrixXd C = Eigen::MatrixXd::Random(d, r);
    if (Context::get_global_tid() == 0) {
        C_initialize_agg.update(C);
    }
    lib::AggregatorFactory::sync();
    C = C_initialize_agg.get_value();
    // initialize code matrix B
    Eigen::MatrixXd raw_B = C.transpose() * mat; 
    Eigen::MatrixXd B = binary_convert(raw_B, r, n);
    for ( int i= 0; i < r; i++) {
        for (int j = 0; j < r; j++){
            if (B(i, j) == 0) {
            //    LOG_I << "Initial B has 0";
            }
        }
    }
    // initialize lagrangian multipliers A
    lib::Aggregator<Eigen::MatrixXd> A_initialize_agg(ZERO_d_r, [](Eigen::MatrixXd& a, const Eigen::MatrixXd& b){ a = b; });
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(d, r);
    if (Context::get_global_tid() == 0) {
        A_initialize_agg.update(A);
    }
    lib::AggregatorFactory::sync();
    A = A_initialize_agg.get_value();

    // store the matrices in the node
    auto & NodeObject_list = ObjListStore::create_objlist<NodeObject>("Nodes");
    NodeObject Node;
    Node.key_init(std::to_string(Context::get_global_tid()));
    Node.X = mat;
    Node.old_C = C;
    Node.new_C = C;
    Node.old_B = B;
    Node.new_B = B;
    Node.A = A;
    Node.p = p;
    Node.name_list = name_list;
    Node.err = 0;
    NodeObject_list.add_object(Node);
    globalize(NodeObject_list);
    // 
    // train the model
    // aggregator to count  the sum of new C's and old C's
    Eigen::MatrixXd C_new_sum(d, r);
    lib::Aggregator<Eigen::MatrixXd> C_new_sum_agg(ZERO_d_r, [&](Eigen::MatrixXd& a, const Eigen::MatrixXd& b){ if(a.cols() == b.cols() && a.rows() == b.rows()) a += b; }, [&](Eigen::MatrixXd& v){v = std::move(Eigen::MatrixXd(d, r));});
    C_new_sum_agg.to_reset_each_iter();
    Eigen::MatrixXd C_old_sum(d, r);
    lib::Aggregator<Eigen::MatrixXd> C_old_sum_agg(ZERO_d_r, [&](Eigen::MatrixXd& a, const Eigen::MatrixXd& b){ if(a.cols() == b.cols() && a.rows() == b.rows()) a += b; }, [&](Eigen::MatrixXd& v){v = std::move(Eigen::MatrixXd(d, r));});
    C_old_sum_agg.to_reset_each_iter();
    

    // count the number of matrices
    int local_node_count = 0;
    lib::Aggregator<int> matrices_number_sum_agg(0, [&](int& a, const int& b){ a += b; });
    list_execute(NodeObject_list, [&](NodeObject & m) {
        matrices_number_sum_agg.update(1);
        local_node_count += 1;
    });
    lib::AggregatorFactory::sync();
    int total_matrices = matrices_number_sum_agg.get_value();

//    LOG_I << "worker " << Context::get_global_tid() << " has " << local_node_count << " node, START";
    int epoch = 0;
    while(epoch < max_epoch) {
        // aggreagator to count the sum of error
        lib::Aggregator<int> err_sum_agg(0, [](int& a, const int& b){ a += b; }, [&](int v){v = std::move(int(0));});
        err_sum_agg.to_reset_each_iter();
        for (int k = 0; k < K; k++) {
//            LOG_I << "iteration " << k;
            list_execute(NodeObject_list, [&](NodeObject & N) {
                N.old_B = N.new_B;
                N.old_C = N.new_C;
                // compute the sum of C 
                C_old_sum_agg.update(N.old_C);
            });
            lib::AggregatorFactory::sync();
            C_old_sum = C_old_sum_agg.get_value();
            list_execute(NodeObject_list, [&](NodeObject & N) {
                Eigen::MatrixXd S(r, d);
                S = 2 * N.old_B * (N.X).transpose() - (N.A).transpose() + N.p * (C_old_sum - N.old_C).transpose();
                // compute G E H
                // we assume r < d
                Eigen::MatrixXd G(r, r);
                Eigen::MatrixXd E(r, d);
                Eigen::MatrixXd H(d, r);
                // G E H = SVD(S);
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::MatrixXd tmp_E = svd.singularValues().asDiagonal();
                for (int i = 0; i < r; i++) {
                    E(i, i) = tmp_E(i, i);
                }
                G = svd.matrixU();
                H = svd.matrixV();
                // update Ci
                N.new_C = H * G.transpose();
                C_new_sum_agg.update(N.new_C);
            });
            lib::AggregatorFactory::sync();
            C_new_sum = C_new_sum_agg.get_value();
            // update Ai
            list_execute(NodeObject_list, [&](NodeObject & N) {
                N.A = N.A + 2 * N.p * total_matrices * N.new_C - 2 * p * C_new_sum;
            });
            lib::AggregatorFactory::sync();
        }
        list_execute(NodeObject_list, [&](NodeObject & N) {
            int n = (N.X).cols();
            Eigen::MatrixXd raw_B(r, n);
            raw_B = (N.new_C).transpose() * N.X;
            Eigen::MatrixXd new_B(r, n);
            new_B = binary_convert(raw_B, r, n);
            N.new_B = new_B;
        });
        epoch += 1;
        if (Context::get_global_tid() == 0) {
            LOG_I << "Epoch" << epoch;
        }

    }
    if (Context::get_global_tid() == 0) {
        LOG_I << "training for the hashing function is done";
    }

    // make all C equal
    lib::Aggregator<Eigen::MatrixXd> C_equal_agg(ZERO_d_r, [&](Eigen::MatrixXd& a, const Eigen::MatrixXd& b){ if(a.cols() == b.cols() && a.rows() == b.rows()) a = b; }, [&](Eigen::MatrixXd& v){v = std::move(Eigen::MatrixXd(d, r));});
    C_equal_agg.to_reset_each_iter();
    list_execute(NodeObject_list, [&](NodeObject& obj) {
        C_equal_agg.update(obj.new_C);
    });
    lib::AggregatorFactory::sync();
    Eigen::MatrixXd final_C(d, r);
    final_C = C_equal_agg.get_value();
    list_execute(NodeObject_list, [&](NodeObject& obj) {
        obj.new_C = final_C;
        obj.old_C = final_C;
    });

    // get the base url
    std::string base_url = Context::get_param("base_url");
    // read the base data 
    husky::io::BinaryInputFormat infmt_base(base_url);
    cur_col = 0;
    n = 0;
    auto & base_data_list = ObjListStore::create_objlist<VectorObject>("data of base");
    husky::load(infmt_base, [&](husky::base::BinStream& file) {
        int size = file.size();
        std::vector<double> row_vector;
        while (file.size()) {
            float sz = husky::base::deser<float>(file);  
            if (cur_col != 0) {
                row_vector.push_back(sz);
            }
            cur_col ++;
            if (cur_col == d+1) {
                std::string key = "base" + std::to_string(n);
                VectorObject X;
                X.key_init(key);
                Eigen::VectorXd x(d);
                for (int k = 0; k < d; k++) {
                    x(k) = row_vector[k];
                }
                X.vec = x;
                base_data_list.add_object(X); 
                row_vector.clear();
                cur_col = 0;
                n++;
            }
        }
        LOG_I << "total " << n << " base data";
    });
    if (Context::get_global_tid() == 0) {
        LOG_I << "reading base fvecs done";
    }
    // centralize the data
    list_execute(base_data_list, [&](VectorObject& obj) {
        obj.vec = obj.vec - Mean_Vector;
    });
    // hash the data
    auto & base_binary_list = ObjListStore::create_objlist<VectorObject>("binary of base");
    list_execute(base_data_list, [&](VectorObject& obj) {
        std::string key = obj.name;
        VectorObject X;
        X.key_init(key);
        Eigen::VectorXd x(r);
        Eigen::MatrixXd tmp_pro(1,r);
        Eigen::MatrixXd tmp_vec(1,d);
        for (int i = 0; i < d; i++) {
            tmp_vec(0, i) = (obj.vec)(i);
        }
        tmp_pro = tmp_vec * final_C;
        for (int i = 0; i < r; i++) {
            x(i) = tmp_pro(0, i);
        }
        X.vec = x;
        base_binary_list.add_object(X);
    });
    // write the base data
    std::string base_binary_str;
    list_execute(base_binary_list, [&](VectorObject& obj) {
        base_binary_str += obj.name;
        base_binary_str += ":";
        for (int i = 0; i < r; i++) {
            if (obj.vec(i) > 0) {
                base_binary_str += "1"; 
            }
            else {
                base_binary_str += "0"; 
            }
        }
        base_binary_str += "\n";
    });
    std::string final_base_binary_url;
    final_base_binary_url = husky::Context::get_param("output_base_binary")+husky::Context::get_param("r")+ 
        "_epoch" + husky::Context::get_param("max_epoch");
    husky::io::HDFS::Write("master", "9000", base_binary_str, 
            final_base_binary_url,
            husky::Context::get_global_tid());
    husky::io::HDFS::CloseFile("master", "9000");
    if (Context::get_global_tid() == 0)
        LOG_I << "have written base_binary " << final_base_binary_url;

    // get the query url
    std::string query_url = Context::get_param("query_url");
    husky::io::BinaryInputFormat infmt_query(query_url);
    cur_col = 0;
    n = 0;
    auto & query_data_list = ObjListStore::create_objlist<VectorObject>("data of query");
    husky::load(infmt_query, [&](husky::base::BinStream& file) {
        int size = file.size();
        std::vector<double> row_vector;
        while (file.size()) {
            float sz = husky::base::deser<float>(file);  
            if (cur_col != 0) {
                row_vector.push_back(sz);
            }
            cur_col ++;
            if (cur_col == d+1) {
                std::string key = "query" + std::to_string(n);
                VectorObject X;
                X.key_init(key);
                Eigen::VectorXd x(d);
                for (int k = 0; k < d; k++) {
                    x(k) = row_vector[k];
                }
                X.vec = x;
                query_data_list.add_object(X); 
                row_vector.clear();
                cur_col = 0;
                n++;
            }
        }
        LOG_I << "total " << n << " query data";
    });
    if (Context::get_global_tid() == 0)
        LOG_I << "reading query fvecs done";
    // centralize the data
    list_execute(query_data_list, [&](VectorObject& obj) {
        obj.vec = obj.vec - Mean_Vector;
    });
    // hash the data
    auto & query_binary_list = ObjListStore::create_objlist<VectorObject>("binary of query");
    list_execute(query_data_list, [&](VectorObject& obj) {
        std::string key = obj.name;
        VectorObject X;
        X.key_init(key);
        Eigen::VectorXd x(r);
        Eigen::MatrixXd tmp_pro(1,r);
        Eigen::MatrixXd tmp_vec(1,d);
        for (int i = 0; i < d; i++) {
            tmp_vec(0, i) = (obj.vec)(i);
        }
        tmp_pro = tmp_vec * final_C;
        for (int i = 0; i < r; i++) {
            x(i) = tmp_pro(0, i);
        }
        X.vec = x;
        query_binary_list.add_object(X);
    });
    std::string query_binary_str;
    list_execute(query_binary_list, [&](VectorObject& obj) {
        query_binary_str += obj.name;
        query_binary_str += ":";
        for (int i = 0; i < r; i++) {
            if (obj.vec(i) > 0) {
                query_binary_str += "1"; 
            }
            else {
                query_binary_str += "0"; 
            }
        }
        query_binary_str += "\n";
    });
    std::string final_query_binary_url;
    final_query_binary_url = husky::Context::get_param("output_query_binary")+husky::Context::get_param("r")+ 
        "_epoch" + husky::Context::get_param("max_epoch");
    husky::io::HDFS::Write("master", "9000", query_binary_str, 
            final_query_binary_url,
            husky::Context::get_global_tid());
    husky::io::HDFS::CloseFile("master", "9000");
    if (Context::get_global_tid() == 0)
        LOG_I << "have written query_binary " << final_query_binary_url;


    
}

} // namespace husky


int main(int argc, char** argv) {
    std::vector<std::string> args(
        {"hdfs_namenode", "hdfs_namenode_port", "max_epoch", "N", "K", "p", "d", "r", 
            "learn_url", "base_url", "query_url", "output_base_binary", "output_query_binary"});
    if (husky::init_with_args(argc, argv, args)) {
        husky::run_job(husky::DISH);
        return 0;
    }
    return 1;
}

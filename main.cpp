#include <iostream>
#include <opencv2/opencv.hpp>

#include "dataset.hpp"
#include "knn.hpp"
#include "normalBayes.hpp"
#include "logisticRegression.hpp"
#include "decisionTree.hpp"
#include "randomForest.hpp"
#include "svm.hpp"
#include "mlp.hpp"


int main()
{
    //Create Dataset
    const char* imagePath = "/home/zana/Documents/imrid/Machine_Learning/digits.png";
    auto dataSet = createTrainData(imagePath);
    dataSet->setTrainTestSplitRatio(0.8,true);

    //Train
    trainKnn(dataSet);
    trainNormalBayes(dataSet);
    trainLogisticRegression(dataSet);
    trainDecisionTree(dataSet);
    trainRandomForests(dataSet);
    trainSVM(dataSet);
    trainMLP(dataSet);

    //Test
    float knnError = testKnn(dataSet);
    float nbError = testNormalBayes(dataSet);
    float lrError = testLogisticRegression(dataSet);
    float dtError = testDecisionTree(dataSet);
    float rfError = testRandomForest(dataSet);
    float svmError = testSVM(dataSet);
    float mlpError = testMLP(dataSet);

    //Accuracy
    std::cout << "KNN error: " << knnError << "%" << std::endl;
    std::cout << "Normal Bayes error: " << nbError << "%" << std::endl;
    std::cout << "Logistic Regression error: " << lrError << "%" << std::endl;
    std::cout << "Decision Tree error: " << dtError << "%" << std::endl;
    std::cout << "Random Forest error: " << rfError << "%" << std::endl;
    std::cout << "SVM error: " << svmError << "%" << std::endl;
    std::cout << "MLP error: " << mlpError << "%" << std::endl;

    return 0;
}

#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainSVM(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto svm = cv::ml::SVM::create();

    svm->setKernel(cv::ml::SVM::LINEAR); //cv::ml::SVM::RBF, cv::ml::SVM::SIGMOID, cv::ml::SVM::POLY
    svm->setType(cv::ml::SVM::C_SVC);

    svm->trainAuto(dataset);

    svm->save("SVM.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testSVM(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto svm = cv::ml::SVM::load("SVM.xml");

    std::vector<int32_t> predictions;
    auto error = svm->calcError(dataset, true, predictions);

    return error;
}

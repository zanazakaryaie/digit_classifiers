#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainNormalBayes(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto normal_bayes = cv::ml::NormalBayesClassifier::create();

    cv::Mat trainData = dataset->getTrainSamples();
    cv::Mat trainLabels = dataset->getTrainResponses();

    normal_bayes->train(trainData, 0, trainLabels);

    normal_bayes->save("NormalBayes.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testNormalBayes(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto normal_bayes = cv::ml::NormalBayesClassifier::load("NormalBayes.xml");

    std::vector<int32_t> predictions;
    auto error = normal_bayes->calcError(dataset, true, predictions);

    return error;
}

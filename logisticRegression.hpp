#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainLogisticRegression(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto logistic_regression = cv::ml::LogisticRegression::create();

    logistic_regression->setLearningRate(0.001);
    logistic_regression->setIterations(1000);
    logistic_regression->setRegularization(cv::ml::LogisticRegression::REG_L2);
    logistic_regression->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
    logistic_regression->setMiniBatchSize(100);

    cv::Mat trainData = dataset->getTrainSamples();
    cv::Mat trainLabels = dataset->getTrainResponses();
    trainLabels.convertTo(trainLabels, CV_32F);

    logistic_regression->train(trainData, 0, trainLabels);

    logistic_regression->save("LogisticRegression.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testLogisticRegression(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto logistic_regression = cv::ml::LogisticRegression::load("LogisticRegression.xml");

    std::vector<int32_t> predictions;
    auto error = logistic_regression->calcError(dataset, true, predictions);

    return error;
}

#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainRandomForests(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto random_forest = cv::ml::RTrees::create();

    random_forest->setMaxCategories(2);
    random_forest->setMaxDepth(20);
    random_forest->setMinSampleCount(1);
    random_forest->setTruncatePrunedTree(true);
    random_forest->setUse1SERule(true);
    random_forest->setUseSurrogates(false);
    random_forest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 5000, 1e-8));
    random_forest->setCVFolds(1);

    random_forest->train(dataset);

    random_forest->save("RandomForest.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testRandomForest(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto random_forest = cv::ml::RTrees::load("RandomForest.xml");

    std::vector<int32_t> predictions;
    auto error = random_forest->calcError(dataset, true, predictions);

    return error;
}

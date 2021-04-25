#pragma once

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void trainDecisionTree(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto decision_tree = cv::ml::DTrees::create();

    decision_tree->setMaxCategories(2);
    decision_tree->setMaxDepth(20);
    decision_tree->setMinSampleCount(1);
    decision_tree->setTruncatePrunedTree(true);
    decision_tree->setUse1SERule(true);
    decision_tree->setUseSurrogates(false);
    decision_tree->setCVFolds(1);

    decision_tree->train(dataset);

    decision_tree->save("DecisionTree.xml");
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float testDecisionTree(const cv::Ptr<cv::ml::TrainData> &dataset)
{
    auto decision_tree = cv::ml::DTrees::load("DecisionTree.xml");

    std::vector<int32_t> predictions;
    auto error = decision_tree->calcError(dataset, true, predictions);

    return error;
}

"""
Title: 
    Table Pretrained Model class

Description:
    This file has the Pretrained model class used in training time.

Takeaways:
    - Since this is pytorch_lightning, we need the same class to test/predict.
    - Can modify it to Huggingface structure to future scope.

Author: purnasai@soulpage
Date: 10-10-2023
"""

import torch
import pandas as pd
import lightning.pytorch as pl

from torchmetrics.classification import F1Score
from torchmetrics import ConfusionMatrix,  Precision
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import logging
logger = logging.getLogger(__name__)


## list of table tags/labels we used at train time in the proper order
labels = ['us-gaap:StockholdersEquity',
 'us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
 'us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
 'us-gaap:NetCashProvidedByUsedInOperatingActivities',
 'us-gaap:LiabilitiesCurrent',
 'us-gaap:AssetsCurrent',
 'us-gaap:NetCashProvidedByUsedInInvestingActivities',
 'us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
 'us-gaap:NetCashProvidedByUsedInFinancingActivities',
 'us-gaap:RetainedEarningsAccumulatedDeficit',
 'us-gaap:ShareBasedCompensation',
 'us-gaap:AccumulatedOtherComprehensiveIncomeLossNetOfTax',
 'us-gaap:NetIncomeLoss',
 'us-gaap:OtherAssetsNoncurrent',
 'us-gaap:CashAndCashEquivalentsAtCarryingValue',
 'us-gaap:OtherLiabilitiesNoncurrent',
 'us-gaap:Assets',
 'us-gaap:LiabilitiesAndStockholdersEquity',
 'us-gaap:PropertyPlantAndEquipmentNet',
 'us-gaap:Goodwill',
 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
 'us-gaap:EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
 'us-gaap:Liabilities',
 'us-gaap:PaymentsForProceedsFromOtherInvestingActivities',
 'us-gaap:CostOfGoodsAndServicesSold',
 'us-gaap:DeferredIncomeTaxAssetsNet',
 'us-gaap:MinorityInterest',
 'us-gaap:InventoryNet',
 'us-gaap:IntangibleAssetsNetExcludingGoodwill',
 'us-gaap:AccountsPayableCurrent',
 'us-gaap:ProfitLoss',
 'us-gaap:DeferredIncomeTaxExpenseBenefit',
 'us-gaap:PaymentsForRepurchaseOfCommonStock',
 'us-gaap:ProceedsFromPaymentsForOtherFinancingActivities',
 'us-gaap:OtherNonoperatingIncomeExpense',
 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
 'us-gaap:OperatingLeaseRightOfUseAsset',
 'us-gaap:LongTermDebtNoncurrent',
 'us-gaap:OperatingLeaseLiabilityNoncurrent',
 'us-gaap:OperatingIncomeLoss',
 'us-gaap:WeightedAverageNumberOfSharesOutstandingBasic',
 'us-gaap:OtherAssetsCurrent',
 'us-gaap:DepreciationDepletionAndAmortization',
 'us-gaap:DepreciationAndAmortization',
 'us-gaap:LongTermDebtAndCapitalLeaseObligations',
 'us-gaap:AdditionalPaidInCapital',
 'us-gaap:IncomeTaxExpenseBenefit',
 'us-gaap:AdditionalPaidInCapitalCommonStock',
 'us-gaap:IncreaseDecreaseInAccountsReceivable',
 'us-gaap:PrepaidExpenseAndOtherAssetsCurrent',
 'us-gaap:DeferredIncomeTaxLiabilitiesNet',
 'us-gaap:IncreaseDecreaseInInventories',
 'us-gaap:OtherOperatingActivitiesCashFlowStatement',
 'us-gaap:DebtCurrent',
 'us-gaap:ProceedsFromStockOptionsExercised',
 'us-gaap:IncreaseDecreaseInAccountsPayableAndAccruedLiabilities',
 'us-gaap:PaymentsRelatedToTaxWithholdingForShareBasedCompensation',
 'us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding',
 'us-gaap:PaymentsOfDividends',
 'us-gaap:AccruedIncomeTaxesCurrent',
 'us-gaap:IncreaseDecreaseInAccountsPayable',
 'us-gaap:PropertyPlantAndEquipmentGross',
 'us-gaap:AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment',
 'us-gaap:EmployeeRelatedLiabilitiesCurrent',
 'us-gaap:IncreaseDecreaseInAccruedIncomeTaxesPayable',
 'us-gaap:IncreaseDecreaseInOtherOperatingCapitalNet',
 'us-gaap:LongTermDebtCurrent',
 'us-gaap:IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets',
 'us-gaap:OperatingExpenses',
 'us-gaap:AccruedLiabilitiesCurrent',
 'us-gaap:ProvisionForDoubtfulAccounts',
 'us-gaap:InterestExpense',
 'us-gaap:AccountsPayableAndAccruedLiabilitiesCurrent',
 'us-gaap:AccountsReceivableNetCurrent',
 'us-gaap:OtherLiabilitiesCurrent',
 'us-gaap:Depreciation',
 'us-gaap:AmortizationOfIntangibleAssets',
 'us-gaap:DeferredIncomeTaxesAndTaxCredits',
 'us-gaap:InterestPaidNet',
 'us-gaap:IncreaseDecreaseInOtherOperatingAssets',
 'us-gaap:ProceedsFromSaleOfPropertyPlantAndEquipment',
 'us-gaap:SellingGeneralAndAdministrativeExpense',
 'us-gaap:ProceedsFromRepaymentsOfShortTermDebt',
 'us-gaap:GrossProfit',
 'us-gaap:RepaymentsOfLongTermDebt',
 'us-gaap:PaymentsToAcquireMarketableSecurities',
 'us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations',
 'us-gaap:OperatingLeaseLiabilityCurrent',
 'us-gaap:PaymentsToAcquireInvestments',
 'us-gaap:PaymentsOfDividendsCommonStock',
 'us-gaap:ShortTermBorrowings',
 'us-gaap:LiabilitiesNoncurrent',
 'us-gaap:GeneralAndAdministrativeExpense',
 'us-gaap:RedeemableNoncontrollingInterestEquityCarryingAmount',
 'us-gaap:ShortTermInvestments',
 'us-gaap:OtherNoncashIncomeExpense',
 'us-gaap:IncreaseDecreaseInDeferredRevenue',
 'us-gaap:IncomeTaxesPaidNet',
 'us-gaap:InterestIncomeExpenseNet',
 'us-gaap:ReceivablesNetCurrent',
 'us-gaap:EquityMethodInvestments',
 'us-gaap:NetIncomeLossAttributableToNoncontrollingInterest',
 'us-gaap:ProceedsFromSaleAndMaturityOfMarketableSecurities',
 'us-gaap:RestrictedCashAndCashEquivalentsAtCarryingValue',
 'us-gaap:IncreaseDecreaseInReceivables',
 'us-gaap:EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations',
 'us-gaap:AssetsNoncurrent',
 'us-gaap:GainLossOnInvestments',
 'us-gaap:IncreaseDecreaseInFinanceReceivables',
 'us-gaap:SellingAndMarketingExpense',
 'us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseExcludingExchangeRateEffect',
 'us-gaap:OperatingLeaseRightOfUseAssetAmortizationExpense',
 'us-gaap:PaymentsToAcquireProductiveAssets',
 'us-gaap:ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions',
 'us-gaap:CapitalExpendituresIncurredButNotYetPaid',
 'us-gaap:ProceedsFromRepaymentsOfShortTermDebtMaturingInThreeMonthsOrLess',
 'us-gaap:IncreaseDecreaseInOtherOperatingLiabilities',
 'us-gaap:NonoperatingIncomeExpense',
 'us-gaap:DeferredRevenueCurrent',
 'us-gaap:TreasuryStockValue',
 'us-gaap:ProceedsFromIssuanceOfLongTermDebt',
 'us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic',
 'us-gaap:TreasuryStockCommonValue',
 'us-gaap:LongTermInvestments',
 'us-gaap:OtherAccruedLiabilitiesCurrent',
 'us-gaap:InvestmentIncomeInterest',
 'us-gaap:IncomeLossFromEquityMethodInvestments',
 'us-gaap:IncreaseDecreaseInEmployeeRelatedLiabilities',
 'us-gaap:ProceedsFromIssuanceOfCommonStock',
 'us-gaap:ResearchAndDevelopmentExpense',
 'us-gaap:InvestmentIncomeInterestAndDividend',
 'us-gaap:GainLossOnSaleOfInvestments',
 'us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
 'us-gaap:GainLossOnSaleOfPropertyPlantEquipment',
 'us-gaap:DefinedBenefitPensionPlanLiabilitiesNoncurrent',
 'us-gaap:OtherPostretirementDefinedBenefitPlanLiabilitiesNoncurrent',
 'us-gaap:PensionAndOtherPostretirementBenefitExpense',
 'us-gaap:IncreaseDecreaseInAccruedTaxesPayable',
 'us-gaap:IncomeLossFromEquityMethodInvestmentsNetOfDividendsOrDistributions',
 'us-gaap:IncreaseDecreaseInAccountsReceivableAndOtherOperatingAssets',
 'us-gaap:IncreaseDecreaseInAccountsPayableAndOtherOperatingLiabilities',
 'us-gaap:ProceedsFromCollectionOfFinanceReceivables',
 'us-gaap:PaymentsToAcquireEquityMethodInvestments',
 'us-gaap:CommonStockValueOutstanding',
 'us-gaap:OperatingCostsAndExpenses',
 'us-gaap:PensionAndOtherPostretirementBenefitContributions',
 'us-gaap:TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests',
 'us-gaap:PensionAndOtherPostretirementBenefitsExpenseReversalOfExpenseNoncash',
 'us-gaap:ProceedsFromSaleMaturityAndCollectionsOfInvestments',
 'us-gaap:RepaymentsOfDebtMaturingInMoreThanThreeMonths',
 'us-gaap:TaxesPayableCurrent',
 'us-gaap:RestrictedCash',
 'us-gaap:Revenues',
 'us-gaap:ForeignCurrencyTransactionGainLossBeforeTax',
 'us-gaap:FairValueAdjustmentOfWarrants',
 'us-gaap:ContractWithCustomerLiabilityCurrent',
 'us-gaap:IncreaseDecreaseInContractWithCustomerLiability',
 'us-gaap:IncomeTaxesPaid',
 'us-gaap:AmortizationOfFinancingCostsAndDiscounts',
 'us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax',
 'us-gaap:OtherOperatingIncomeExpenseNet',
 'us-gaap:OtherInvestments',
 'us-gaap:IncreaseDecreaseInOperatingCapital',
 'us-gaap:RepaymentsOfDebt',
 'us-gaap:RestrictedCashAndCashEquivalents',
 'us-gaap:IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest',
 'us-gaap:ProceedsFromRepaymentsOfCommercialPaper',
 'us-gaap:NetCashProvidedByUsedInFinancingActivitiesContinuingOperations',
 'us-gaap:AdjustmentsToAdditionalPaidInCapitalSharebasedCompensationRequisiteServicePeriodRecognitionValue',
 'us-gaap:AdjustmentsToAdditionalPaidInCapitalOther',
 'us-gaap:TreasuryStockValueAcquiredCostMethod',
 'us-gaap:Dividends',
 'us-gaap:OtherComprehensiveIncomeLossForeignCurrencyTransactionAndTranslationAdjustmentBeforeTax',
 'us-gaap:OtherComprehensiveIncomeLossCashFlowHedgeGainLossAfterReclassificationBeforeTax',
 'us-gaap:OtherComprehensiveIncomeLossTax',
 'us-gaap:OtherDepreciationAndAmortization',
 'us-gaap:IncreaseDecreaseInAccruedLiabilitiesAndOtherOperatingLiabilities',
 'us-gaap:LiabilitiesOtherThanLongtermDebtNoncurrent',
 'us-gaap:ProvisionForLoanLeaseAndOtherLosses',
 'us-gaap:CostsAndExpenses']

label2id = {lable:idx for idx,lable in enumerate(labels)}
id2label = {index:label for label,index in label2id.items()}


class NameMappingModel(pl.LightningModule):
    """Class for ML model realted to Tables, 
    this is a pytorch lightning class. this needs
    forward, training_step, validation_step and test_step
    functions.
    
    This is the exact class used for training as well. same class is
    initiated once again to load trained model."""
    def __init__(self, labels, label2id, id2label):
        super().__init__()
        self.all_test_labels = []
        self.all_test_preds = []
        self.labels = labels
        self.finbert = AutoModelForSequenceClassification.from_pretrained("soleimanian/financial-roberta-large-sentiment",
                                                                        num_labels=len(labels),
                                                                        # problem_type="multi_class_classification",
                                                                        label2id = label2id,
                                                                        id2label  = id2label,
                                                                        ignore_mismatched_sizes=True)
        self.f1 = F1Score(task="multiclass", num_classes= len(labels))
        self.Conf_matrix  = ConfusionMatrix(task="multiclass", num_classes=len(labels))
        self.macro_precision = Precision(task="multiclass", average='macro', num_classes= len(labels))
        self.micro_precision = Precision(task="multiclass", average='micro', num_classes= len(labels))
        # precion-recall curve for multiclass doesn't plot well in the figure, so avoiding it.


    def forward(self, x, y):
        output = self.finbert(**x, labels = y)
        return output

    def training_step(self, batch, batch_idx):
        data,labels  = batch
        # outputs = self.finbert(**data, labels = labels)
        outputs = self.forward(x =data, y = labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        train_f1_score = self.f1(preds, labels)
        train_macro_precision = self.macro_precision(preds,labels)
        train_micro_precision = self.micro_precision(preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", train_f1_score)
        self.log("train_macro_precision", train_macro_precision)
        self.log("train_micro_precision", train_micro_precision)
        return loss

    def validation_step(self, batch, batch_idx):
        data,labels  = batch
        # outputs = self.finbert(**data, labels = labels)
        outputs = self.forward(x =data, y=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_f1_score = self.f1(preds, labels)
        val_macro_precision = self.macro_precision(preds,labels)
        val_micro_precision = self.micro_precision(preds, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1", val_f1_score)
        self.log("val_macro_precision", val_macro_precision)
        self.log("val_micro_precision", val_micro_precision)
        return loss

    def test_step(self, batch, batch_idx):
        data,labels  = batch
        # outputs = self.finbert(**data, labels = labels)
        outputs = self.forward(x=data, y=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        self.all_test_labels.append(labels)
        self.all_test_preds.append(preds)

        test_f1_score = self.f1(preds, labels)

        test_macro_precision = self.macro_precision(preds,labels)
        test_micro_precision = self.micro_precision(preds, labels)


        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_f1", test_f1_score)
        self.log("test_macro_precision", test_macro_precision)
        self.log("test_micro_precision", test_micro_precision)
        return loss

    def on_test_epoch_end(self):
        labels = torch.cat(self.all_test_labels)
        preds = torch.cat(self.all_test_preds)

        test_final_f1_score = self.f1(preds, labels)
        self.log("test_final_f1_score:",test_final_f1_score)

        conf_mat = self.Conf_matrix(preds, labels)
        computed_confusion = conf_mat.detach().cpu().numpy().astype(int)
        df_cm = pd.DataFrame(
            computed_confusion,
            index=self.labels,
            columns=self.labels)
        df_cm.to_excel("confusion_matrix.xlsx")

    def predict_step(self, batch, batch_idx):
      # this is acting same as test_step in our case
      # we dont need this anymore
        data, label = batch
        outputs = self.forward(x=data, y=label)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        result = [id2label[pred_label.item()] for pred_label in predicted_labels]
        return result


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modeleval = NameMappingModel.load_from_checkpoint(checkpoint_path="Models/Table_Inline_Model/sample-sectag-epoch=23-val_loss=0.19.ckpt",
                                                    labels=labels,
                                                    label2id = label2id,
                                                    id2label=id2label,
                                                    map_location= device,
                                                    strict=False)
# disable randomness, dropout, etc...
modeleval = modeleval.eval()
tokenizer = AutoTokenizer.from_pretrained("soleimanian/financial-roberta-large-sentiment")

def predict_table_tags(data):
    logger.info("2.4. Predicting table tags......")
    texts = []
    predicted_labels = []
    # predict with the model
    with torch.no_grad():
        for table_data in data:
            for row in table_data:
                text, tag = row.split("==")
                inputs = tokenizer(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
                
                # random tag just to pass to model to match with syntax
                tag = "us-gaap:StockholdersEquity"
                y = label2id[tag]
                y = torch.tensor(y).squeeze()

                for key in inputs:
                    inputs[key] = inputs[key].to(device)

                outputs = modeleval(inputs, y)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                # truelabels = [id2label[label.item()] for label in labels]
                predlabels = [id2label[label.item()] for label in preds][0]

                texts.append(text)
                predicted_labels.append(predlabels)
    return texts, predicted_labels
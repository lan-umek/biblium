# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:08:33 2025

@author: Lan.Umek
"""
import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from bibgroup import BiblioGroup

class BiblioGroupClassifier(BiblioGroup):
    """
    Extended classifier for predictive and statistical analysis of bibliometric subgroup memberships.

    Inherits from BiblioGroup, adding methods for:
      - Multi-model classification (scikit-learn)
      - Model evaluation and result export
      - Logistic regression analysis (statsmodels)
      - Export of logistic regression results with significance highlighting

    Attributes:
        features_columns (list): Column names from self.df used for tabular features.
        text_columns (list): Column names from self.df used for text-based features.
        max_tfidf_features (int): Max tokens for TF-IDF vectorizer.
        max_count_features (int): Max tokens for CountVectorizer.
        features_df (pd.DataFrame): DataFrame holding selected feature columns.
    """

    def __init__(self, *args,
                 features_columns=None,
                 text_columns=None,
                 max_tfidf_features=500,
                 max_count_features=500,
                 **kwargs):
        """
        Initialize BiblioGroupClassifier.

        Args:
            *args: forwarded to super().__init__
            features_columns (list, optional): numeric feature columns
            text_columns (list, optional): text columns for classification
            max_tfidf_features (int): for TfidfVectorizer
            max_count_features (int): for CountVectorizer (logistic)
            **kwargs: forwarded to super().__init__
        """
        super().__init__(*args, **kwargs)
        self.features_columns = features_columns or []
        self.text_columns = text_columns or []
        self.max_tfidf_features = max_tfidf_features
        self.max_count_features = max_count_features
        self.features_df = pd.DataFrame()

    def prepare_features(self, features_columns=None):
        """
        Populate self.features_df selecting columns from self.df.

        Args:
            features_columns (list, optional): override default columns
        """
        cols = features_columns or self.features_columns
        if cols:
            self.features_df = self.df[cols]

    def evaluate_classifier(self, X, y, clf, method="cross_validation", cv=5):
        """
        Evaluate a classifier using accuracy, AUC, precision, recall, F1.

        Args:
            X, y: data
            clf: estimator
            method: "cross_validation"|"leave_one_out"|"train_test"
            cv: folds for CV
        Returns:
            dict of metrics
        """
        metrics = ["accuracy","roc_auc","precision","recall","f1"]
        results = {}
        if method in ("cross_validation","leave_one_out"):
            cv_strategy = LeaveOneOut() if method=="leave_one_out" else cv
            for m in metrics:
                scores = cross_val_score(clf,X,y,cv=cv_strategy,scoring=m)
                results[m] = np.mean(scores)
        else:
            Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=0)
            clf.fit(Xtr,ytr)
            preds = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:,1]
            results = {
                "accuracy":accuracy_score(yte,preds),
                "roc_auc":roc_auc_score(yte,proba),
                "precision":precision_score(yte,preds),
                "recall":recall_score(yte,preds),
                "f1":f1_score(yte,preds)
            }
        return results

    def classify_groups(self, classifiers=None,
                        method="cross_validation",
                        multilabel=False,
                        save_results=False,
                        file_prefix="results"):
        """
        Classify each group (or multilabel) with multiple models.

        Args:
            classifiers (dict): name->estimator
            method: eval method
            multilabel: if True, train OneVsRest
            save_results: bool
            file_prefix: excel prefix
        Returns:
            dict: group->model->metrics or 'multilabel'
        """
        if classifiers is None:
            classifiers = {
                "Logistic":LogisticRegression(max_iter=1000),
                "RandomForest":RandomForestClassifier(),
                "GBM":GradientBoostingClassifier(),
                "NaiveBayes":MultinomialNB(),
                "SVM":SVC(probability=True)
            }
        # prepare X
        if self.text_columns:
            tf = TfidfVectorizer(max_features=self.max_tfidf_features)
            X = tf.fit_transform(
                self.df[self.text_columns].agg(" ".join,axis=1)
            )
            X = pd.DataFrame(X.toarray(),columns=tf.get_feature_names_out())
        else:
            self.prepare_features()
            X = self.features_df
        results={}
        if multilabel:
            for name,clf in classifiers.items():
                ov = OneVsRestClassifier(clf)
                metrics = self.evaluate_classifier(X,self.group_matrix.values,ov,method)
                results[name]=metrics
        else:
            for grp in self.group_matrix.columns:
                y = self.group_matrix[grp]
                results[grp]={}
                for name,clf in classifiers.items():
                    results[grp][name]=self.evaluate_classifier(X,y,clf,method)
        if save_results:
            self._save_performance(results,f"{file_prefix}.xlsx")
        return results

    def _save_performance(self, results, fname):
        """
        Save classification performance to Excel.
        """
        rows=[]
        for grp,val in results.items():
            if isinstance(val,dict):
                for m,res in val.items():
                    rows.append({"Group":grp,"Model":m,**res})
            else:
                rows.append({"Model":"multilabel",**val})
        df=pd.DataFrame(rows)
        df.to_excel(fname,index=False)

    def train_classifier(self, clf, multilabel=False):
        """
        Train and return prediction functions.

        Args:
            clf: single estimator
            multilabel: if True, OneVsRest
        Returns:
            dict or function
        """
        # build X same as classify
        if self.text_columns:
            tf = TfidfVectorizer(max_features=self.max_tfidf_features)
            X = tf.fit_transform(
                self.df[self.text_columns].agg(" ".join,axis=1)
            )
            X = pd.DataFrame(X.toarray(),columns=tf.get_feature_names_out())
        else:
            self.prepare_features()
            X = self.features_df
        if multilabel:
            ov = OneVsRestClassifier(clf)
            ov.fit(X,self.group_matrix.values)
            return lambda new_df: ov.predict(TfidfVectorizer(vocabulary=X.columns).transform(
                new_df[self.text_columns].agg(" ".join,axis=1)
            ))
        funcs={}
        for grp in self.group_matrix.columns:
            y=self.group_matrix[grp]
            model=clf.__class__(**clf.get_params())
            model.fit(X,y)
            funcs[grp]=lambda df,mdl=model: mdl.predict(
                pd.DataFrame(TfidfVectorizer(vocabulary=X.columns).transform(
                    df[self.text_columns].agg(" ".join,axis=1)
                ).toarray(),columns=X.columns)
            )
        return funcs

    def logistic_regression_analysis(self, text_column="Abstract",
                                     include_regex=None,
                                     exclude_regex=None,
                                     top_n=50):
        """
        Perform logistic regression per group using CountVectorizer features.

        Args as before.
        """
        texts = self.df[text_column].fillna("")
        vec = CountVectorizer(max_features=self.max_count_features,
                              stop_words='english',binary=True)
        Xc = vec.fit_transform(texts)
        docc=(Xc>0).sum(axis=0).A1
        df_items=pd.DataFrame({"item":vec.get_feature_names_out(),"doc_count":docc})
        if include_regex: df_items=df_items[df_items.item.str.contains(include_regex)]
        if exclude_regex: df_items=df_items[~df_items.item.str.contains(exclude_regex)]
        items=df_items.sort_values('doc_count',ascending=False).head(top_n).item.tolist()
        X=pd.DataFrame(Xc.toarray(),columns=vec.get_feature_names_out())[items]
        X=sm.add_constant(X)
        results={}
        for grp in self.group_matrix.columns:
            y=self.group_matrix[grp]
            try:
                m=sm.Logit(y,X).fit(disp=False)
                results[grp]={'model':m,'summary':m.summary2().tables[1]}
            except np.linalg.LinAlgError:
                print(f"Singular matrix for {grp}")
        return results

    def save_logistic_results(self, results, filename="logistic_results.xlsx"):
        """
        Export logistic regression results to Excel with highlighted p-values.
        """
        with pd.ExcelWriter(filename,engine='xlsxwriter') as w:
            wb=w.book
            fmts={
                0.001:wb.add_format({'bg_color':'#006400','font_color':'#FFF'}),
                0.01: wb.add_format({'bg_color':'#228B22','font_color':'#FFF'}),
                0.05: wb.add_format({'bg_color':'#66CDAA','font_color':'#000'}),
                0.1: wb.add_format({'bg_color':'#98FB98','font_color':'#000'})
            }
            allc=[]
            for grp,data in results.items():
                sumt=data['summary'].copy()
                sumt['OR']=np.exp(sumt['Coef.'])
                sheetf=f"coefficients {grp}"
                sumt.to_excel(w,sheet_name=sheetf,index=True)
                ws=w.sheets[sheetf]
                pidx=sumt.columns.get_loc('P>|z|')+1
                for ri,pv in enumerate(sumt['P>|z|'],start=1):
                    for th,fmt in fmts.items():
                        if pv<=th: ws.write(ri,pidx,pv,fmt);break
                stats=pd.DataFrame({
                    'AIC':[data['model'].aic],
                    'BIC':[data['model'].bic],
                    'Pseudo R-squared':[data['model'].prsquared]
                })
                stats.to_excel(w,sheet_name=f"statistics {grp}",index=False)
                tmp=sumt[['Coef.','OR','P>|z|']].copy()
                tmp.columns=pd.MultiIndex.from_product([[grp],tmp.columns])
                allc.append(tmp)
            comb=pd.concat(allc,axis=1).dropna(how='all')
            comb.to_excel(w,sheet_name='Combined_Coefficients',index=True)
            ws2=w.sheets['Combined_Coefficients']
            for grp in results:
                pcol=(grp,'P>|z|')
                if pcol in comb.columns:
                    cidx=comb.columns.get_loc(pcol)+1
                    for ri,pv in enumerate(comb[pcol],start=1):
                        if pd.isna(pv): continue
                        for th,fmt in fmts.items():
                            if pv<=th: ws2.write(ri,cidx,pv,fmt);break
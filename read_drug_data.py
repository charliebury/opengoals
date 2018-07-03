
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from progress import progress
import pickle

CHEMBL_DRUG_DATA = './evs_chembl-edit.tsv'
PPI_DATA = './untitled.txt'
REPURPOSED_DRUGS = './drug_repurposing_curated-edit.tsv'
GENE_DISEASE_RELATIONS = './matrix_all_18.06.csv'
INTACT_DATA = './intact.txt'
REACTASSOC_DATA = './Reactassoc2.tsv'


class drugs_to_disease():

    def __init__(self, process=False):

        if process:
            self.read_gene_disease_relations()

            # get unique indexing for genes and diseases
            self.gene_ids = {gene: ind for ind, gene in enumerate(sorted(set(self.genes_full)))}
            self.disease_ids = {disease: ind for ind, disease in enumerate(sorted(set(self.diseases_full)))}

            with open('gene_ids.pickle', 'wb') as f:
                pickle.dump(self.gene_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open('disease_ids.pickle', 'wb') as f:
                pickle.dump(self.disease_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

            csv_diseases = open('disease_ids.csv', 'w')
            for val, lab in self.disease_ids.items():
                csv_diseases.write('{},{}\n'.format(lab, val))
            csv_diseases.close()

            self.read_chembl_drugs()

            with open('drug_ids.pickle', 'wb') as f:
                pickle.dump(self.drug_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

            csv_drugs = open('drug_ids.csv', 'w')
            for val, lab in self.drug_ids.items():
                csv_drugs.write('{},{}\n'.format(lab, val))
            csv_drugs.close()

            self.make_drugs_vs_genes_array()
            self.make_genes_vs_disease_array()
            self.multiply_arrays()
            self.drugs_straight_to_diseases()
        else:
            self.retrieve_arrays()
            self.compare_two_arrays()
            # self.add_intact_relations()
            # self.add_reactassoc_relations()
            self.multiply_arrays(save_pickle=False, operation='mult')
            self.compare_two_arrays()

            # save the final array at the end of a run
            with open('drugs_vs_disease_mult-after-PPIandPathways.pickle', 'wb') as f:
                pickle.dump(self.drugs_disease_mult_array, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

        # self.read_repurposed_drug_set()

    def read_chembl_drugs(self, do_prints=False):

        print('Reading chembl drug file')

        cols = ['disease.id', 'disease.efo_info.label', 'drug.molecule_name', 'drug.molecule_type',
                'drug.id', 'evidence.drug2clinic.max_phase_for_disease.numeric_index',
                'evidence.target2drug.action_type', 'evidence.target2drug.mechanism_of_action',
                'target.id', 'target.gene_info.symbol', 'random hash code']

        df = pd.read_csv(CHEMBL_DRUG_DATA, header=None, sep='\t', names=cols)

        unique_drugs = list(set(df['drug.molecule_name'].values))
        unique_diseases = list(set(df['disease.id'].values))
        unique_genes = list(set(df['target.id'].values))
        max_clinic_stage = df['evidence.drug2clinic.max_phase_for_disease.numeric_index'].values
        molecule_types = df['drug.molecule_type'].values
        drug_actions = df['evidence.target2drug.action_type'].values

        if do_prints:
            print('Num uniq drugs:', len(unique_drugs))
            print('Num uniq diseases:', len(unique_diseases))
            print('Num uniq genes:', len(unique_genes))
            print('Molecule type breakdown:', Counter(molecule_types))
            print('Clinic stage breakdown:', Counter(max_clinic_stage))
            print('Drug action breakdown:', Counter(drug_actions))

        drugs = df['drug.id'].values
        genes = df['target.id'].values
        diseases = df['disease.id'].values
        drug_names = df['drug.molecule_name'].values

        # get lists into correct format
        drugs = [str(d).split('/')[-1] for d in drugs]
        genes = [str(g) for g in genes]
        diseases = [str(d) for d in diseases]

        self.drugName_to_chemblID = {name: cid for name, cid in zip(drug_names, drugs)}

        # get a conserved ordering schemes
        self.drug_ids = {drug: ind for ind, drug in enumerate(sorted(set(drugs)))}

        self.drugs = drugs
        self.genes_with_drugs = genes
        self.diseases_with_drugs = diseases
        self.max_clinic_stage = df['evidence.drug2clinic.max_phase_for_disease.numeric_index'].values

    def read_gene_disease_relations(self):

        print('Reading gene disease relation file')
        df = pd.read_csv(GENE_DISEASE_RELATIONS, sep=',')

        genes = df['ensembl_gene_id'].values
        diseases = df['disease_id'].values
        overall_relation_scores = df['overall_score'].values

        # get lists into correct format
        genes = [str(g) for g in genes]
        diseases = [str(d) for d in diseases]

        self.genes_full = genes
        self.diseases_full = diseases
        self.overall_relation_scores_full = overall_relation_scores

    def read_adverse_effects(self):

        cols = ['Drug name', 'Drug ChEMBL ID', 'Adverse event', 'Adverse event count']
        df = pd.read_csv('./drug_adverse_events.tsv', header=None, sep='\t', names=cols)

        side_effects = df['Adverse event'].values
        print('side effects breakdown:', Counter(side_effects))

    def read_repurposing(self):

        cols = ['Drug name', 'Drug ChEMBL ID', 'Adverse event', 'Adverse event count']
        df = pd.read_csv('./drug_adverse_events.tsv', header=None, sep='\t', names=cols)

        side_effects = df['Adverse event'].values
        print('side effects breakdown:', Counter(side_effects))

    def make_drugs_vs_genes_array(self, plot_heatmap=False):

        print('making drug vs genes array')
        dg_arr = np.zeros((len(self.drug_ids.keys()), len(self.gene_ids.keys())))

        tot_count = len(self.drugs)
        for i, (drug, gene) in enumerate(zip(self.drugs, self.genes_with_drugs)):
            progress(i, tot_count)
            if np.isnan(self.max_clinic_stage[i]):
                print(drug, gene)
            dg_arr[self.drug_ids[drug], self.gene_ids[gene]] = 1#self.max_clinic_stage[i]

        if plot_heatmap:
            fig = plt.figure()
            sns.heatmap(dg_arr)
            fig.suptitle('drug vs gene array', fontsize=14)
            plt.xlabel('gene id', fontsize=12)
            plt.ylabel('drug id', fontsize=12)
            fig.savefig('./drugsVsGenes.png')

        self.drugs_genes_array = dg_arr

        with open('drugs_vs_genes.pickle', 'wb') as f:
            pickle.dump(dg_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savetxt('drugs_vs_genes.csv', dg_arr, delimiter=',')

    def make_genes_vs_disease_array(self, plot_heatmap=False):

        print('making genes vs disease array')
        dg_arr = np.zeros((len(self.gene_ids.keys()), len(self.disease_ids.keys())))

        tot_count = len(self.genes_full)
        for i, (gene, disease) in enumerate(zip(self.genes_full, self.diseases_full)):
            progress(i, tot_count)
            dg_arr[self.gene_ids[gene], self.disease_ids[disease]] = self.overall_relation_scores_full[i]

        if plot_heatmap:
            fig = plt.figure()
            sns.heatmap(dg_arr)
            fig.suptitle('genes vs diseases array', fontsize=14)
            plt.xlabel('gene id', fontsize=12)
            plt.ylabel('disease id', fontsize=12)
            fig.savefig('./genesVsDiseases.png')

        self.genes_vs_disease_array = dg_arr

        with open('genes_vs_diseases.pickle', 'wb') as f:
            pickle.dump(dg_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savetxt('genes_vs_diseases.csv', dg_arr, delimiter=',')

    def drugs_straight_to_diseases(self, plot_heatmap=True):

        drug_disease_arr = np.zeros((len(self.drug_ids), len(self.disease_ids)))
        tot_count = len(self.drugs)
        for i, (drug, disease) in enumerate(zip(self.drugs, self.diseases_with_drugs)):
            progress(i, tot_count)
            drug_disease_arr[self.drug_ids[drug], self.disease_ids[disease]] = 1

        self.drugs_straight_to_diseases_array = drug_disease_arr

        with open('drugs_diseases_chembl.pickle', 'wb') as f:
            pickle.dump(drug_disease_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savetxt('drugs_diseases_chembl.csv', drug_disease_arr, delimiter=',')

        if plot_heatmap:
            fig = plt.figure()
            sns.heatmap(drug_disease_arr)
            fig.suptitle('drugs vs diseases array', fontsize=14)
            plt.xlabel('drug id', fontsize=12)
            plt.ylabel('disease id', fontsize=12)
            fig.savefig('./drugsVsDiseases-chembl.png')

    def multiply_arrays(self, plot_heatmap=True, save_pickle=True, operation='mult'):

        if operation == 'mult':
            drugsVdisease = np.matmul(self.drugs_genes_array, self.genes_vs_disease_array)

            # find number of genes linked in chembl per drug
            num_genes_per_drug = np.sum(self.drugs_genes_array, axis=1)

            # divide by number of genes per drug to compute an average drug-disease
            # score across a range of intermediate genes
            drugsVdisease = drugsVdisease/num_genes_per_drug[:, None]

        elif operation == 'max':
            drugsVdisease = np.zeros((self.drugs_genes_array.shape[0], self.genes_vs_disease_array.shape[1]))
            tot_count = self.drugs_genes_array.shape[0]
            for i, drug_id in enumerate(range(self.drugs_genes_array.shape[0])):
                progress(i, tot_count)
                drug_to_genes = self.drugs_genes_array[drug_id, :]
                for disease_id in range(self.genes_vs_disease_array.shape[1]):
                    genes_to_disease = self.genes_vs_disease_array[:, disease_id]
                    max_score = np.max(genes_to_disease*drug_to_genes)
                    drugsVdisease[drug_id, disease_id] = max_score

        if plot_heatmap:
            fig = plt.figure()
            sns.heatmap(drugsVdisease)
            fig.suptitle('drugs vs diseases array', fontsize=14)
            plt.xlabel('disease id', fontsize=12)
            plt.ylabel('drug id', fontsize=12)
            fig.savefig('./drugsVsDiseases-{}.png'.format(operation))

            fig = plt.figure()
            sns.heatmap(self.drugs_straight_to_diseases_array)
            fig.suptitle('drugs vs diseases array', fontsize=14)
            plt.xlabel('disease id', fontsize=12)
            plt.ylabel('drug id', fontsize=12)
            fig.savefig('./drugsVsDiseases-chembl.png')

        if save_pickle:
            with open('drugs_vs_disease_{}.pickle'.format(operation), 'wb') as f:
                pickle.dump(drugsVdisease, f, protocol=pickle.HIGHEST_PROTOCOL)
            np.savetxt('drugs_vs_disease_{}.csv'.format(operation), drugsVdisease, delimiter=',')

        if operation == 'mult':
            self.drugs_disease_mult_array = drugsVdisease
        elif operation == 'max':
            self.drugs_disease_max_array = drugsVdisease

    def retrieve_arrays(self):

        print('extracting data from pickle files')

        with open('drugs_vs_disease_mult.pickle', 'rb') as f:
            self.drugs_disease_mult_array = pickle.load(f)

        with open('drugs_diseases_chembl.pickle', 'rb') as f:
            self.drugs_straight_to_diseases_array = pickle.load(f)

        with open('drugs_vs_genes.pickle', 'rb') as f:
            self.drugs_genes_array = pickle.load(f)

        with open('genes_vs_diseases.pickle', 'rb') as f:
            self.genes_vs_disease_array = pickle.load(f)

        with open('gene_ids.pickle', 'rb') as f:
            self.gene_ids = pickle.load(f)

        with open('disease_ids.pickle', 'rb') as f:
            self.disease_ids = pickle.load(f)

        with open('drug_ids.pickle', 'rb') as f:
            self.drug_ids = pickle.load(f)

    def compare_two_arrays(self, threshold_chembl=0, threshold_mult=1, print_text=False):

        a = self.drugs_straight_to_diseases_array
        b = self.drugs_disease_mult_array
        a_threshed = (a > threshold_chembl).astype(int)
        b_threshed = (b > threshold_mult).astype(int)

        drugIds_to_names = {v: k for k, v in self.drug_ids.items()}

        all_hits_count = 0
        for i in range(a_threshed.shape[0]):
            a_row = a_threshed[i]
            b_row = b_threshed[i]

            comb_row = a_row + b_row
            num_hits = np.sum((comb_row > 1).astype(int))

            gene_num_reported_in_chembl = np.sum(a_row)
            predicted_gene_nums_mult = np.sum(b_row)

            if num_hits != gene_num_reported_in_chembl:
                if print_text:
                    print(drugIds_to_names[i])
                    print('{}, {}, {}'.format(
                        num_hits, gene_num_reported_in_chembl, predicted_gene_nums_mult))
            else:
                all_hits_count += 1

        print('All known drug-diseases found: {}/{}'.format(
            all_hits_count, a_threshed.shape[0]))

    def add_intact_relations(self, weight=0.01):
        """
        add the intact relationships between genes.
        Currently, if a relationship if found between gene A and gene B,
        then the gene-to-disease score for gene B is updated to include the
        average of the original gene B score with the score of gene A (the relative
        contribution of this average relative to the original B score is determined
        by the provided confidence value that gene A and B are similar.
        If the confidence score is 0 between A and B, this would leave B with its
        original value, a score of 1 would make the new B score the full average of
        A and B's original scores
        """

        df = pd.read_csv(INTACT_DATA, sep='\t')
        gene_A = df['Gene A'].values
        gene_B = df['Gene B'].values
        pair_confidence = df['Confidence'].values

        not_found = 0
        for i, (g_a, g_b) in enumerate(zip(gene_A, gene_B)):
            if g_a in self.gene_ids.keys() and g_b in self.gene_ids.keys():
                g_a_id = self.gene_ids[g_a]
                g_b_id = self.gene_ids[g_b]
                contrib_from_a = self.genes_vs_disease_array[g_a_id, :]
                original_b_score = self.genes_vs_disease_array[g_b_id, :]
                average_score = np.mean([contrib_from_a, original_b_score], axis=0)
                weighting = weight*pair_confidence[i]
                self.genes_vs_disease_array[g_b_id, :] = average_score*weighting + (1-weighting)*original_b_score
            else:
                not_found += 1
        print('# gene pairs not found: ', not_found)

    def add_reactassoc_relations(self, weight=0.1):

        f_in = open(REACTASSOC_DATA, 'r')
        contents = f_in.readlines()
        f_in.close()
        tot_count = len(contents)
        for i, ln in enumerate(contents):
            progress(i, tot_count)
            ln = ln.replace('\n', '')
            primary_gene = ln.split('\t')[0]
            other_genes = ln.split('\t')[1:]
            try:
                prim_gene_id = self.gene_ids[primary_gene]
            except KeyError:
                print('Primary gene not found.. skipping')
                continue
            contrib_from_primary = self.genes_vs_disease_array[prim_gene_id, :]
            for j, other_gene in enumerate(other_genes):
                if j > 2:
                    break
                try:
                    oth_gene_id = self.gene_ids[other_gene]
                except KeyError:
                    continue
                orig_oth_score = self.genes_vs_disease_array[oth_gene_id, :]
                average_score = np.mean([contrib_from_primary, orig_oth_score], axis=0)
                self.genes_vs_disease_array[oth_gene_id, :] = average_score*weight + (1-weight)*orig_oth_score


    # def read_repurposed_drug_set(self):

    #     df = pd.read_csv(REPURPOSED_DRUGS, sep='\t', encoding='iso-8859-1')

    #     drug_names = df['drug_name'].values
    #     original_disease = df['original_indication'].values
    #     repurpose = df['repurposed_indication'].values

    #     print(len(drug_names))
    #     drug_ids_chembl = []
    #     for name in drug_names:
    #         try:
    #             chembl_id = self.drugName_to_chemblID[name.upper().strip()]
    #             drug_ids_chembl.append(chembl_id)
    #         except KeyError:
    #             print(name)
    #             pass
    #     print(len(chembl_id))




if __name__ == "__main__":

    drugs_to_disease()

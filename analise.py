#########################################################################################################################
#Imports
import csv 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
#import gower
#########################################################################################################################
#Vars
path = '' #define path here to main folder
ano = '2018' #define year to analyse
p100matri=0
p100tranca=0
p100des=0
p100forma=0
p100trans=0
interesses1=[]
interesses2=[]
cursos = []
#########################################################################################################################
#Funcoes
def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions
    return(chi2_contingency(ctsum.fillna(0)))

def generate_graph_percents(tp,nomes,arq,df):
    lista = []
    ordenado =  df.unique()
    ordenado.sort()
    aux = ordenado -1
    grupos = []
    if tp==0:
        grupos = [nomes[i] for i in (ordenado-2)]
    else:
        aux[len(aux)-1]=aux[len(aux)-1] -1
        grupos = [nomes[i] for i in (aux)]
    file1 = open((path+ano+'/'+arq+'.txt'),"w")
    cont= 0
    for i in ordenado :
        aux = np.count_nonzero(df==i,axis=0)/len(df) *100
        lista.append(aux)
        file1.write(nomes[cont]+':'+str(aux)+'\n')
        cont= cont+1
    y_pos = np.arange(len(ordenado))
    fig, ax = plt.subplots()
    ax.barh(y_pos, lista, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(grupos)
    ax.axes.set_xlim([0,50])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.title.set_text(ano)
    plt.savefig(path+ano+'/'+arq+'.png',bbox_inches = "tight")   # save the figure to file
    plt.clf()     # close the figure window
    file1.close()
    
def generate_graph_avg(arq,df):
    lista = []
    file1 = open((path+ano+'/'+arq+'.txt'),"w")
    media = np.mean(df)
    std = np.std(df)
    median = np.median(df)
    maxim = np.max(df)
    minim = np.min(df)
    file1.write('media:'+str(media)+'\n')
    file1.write('std:'+str(std)+'\n')
    file1.write('mediana:'+str(median)+'\n')
    file1.write('maximo:'+str(maxim)+'\n')
    file1.write('minimo:'+str(minim)+'\n')
    nomes = ['Média','Mediana','Mínimo','Máximo']
    lista = [media,median,minim,maxim]
    y_pos = np.arange(len(nomes))
    fig, ax = plt.subplots()
    ax.barh(y_pos, lista,xerr=std, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nomes)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Valores')
    plt.savefig(path+ano+'/'+arq+'.png',bbox_inches = "tight")   # save the figure to file
    plt.clf()     # close the figure window
    file1.close()
        
def cramers_v(x, y):
        """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
#########################################################################################################################
#Análise
busca = pd.read_csv(path+ano+'/geral_filtro.csv')
#Como as variáveis mudam de nome, as variáveis de interesse entram aqui
if ano=='2017' or ano=='2018':
    if(ano == '2018'):
        interesses1 = ['TP_SITUACAO','TP_CATEGORIA_ADMINISTRATIVA', 'TP_TURNO','TP_MODALIDADE_ENSINO','TP_SEXO','TP_NACIONALIDADE','IN_DEFICIENCIA']
    else:    
        interesses1 = ['TP_SITUACAO','TP_CATEGORIA_ADMINISTRATIVA', 'TP_TURNO','TP_MODALIDADE_ENSINO','TP_SEXO','TP_NACIONALIDADE','TP_DEFICIENCIA']
    interesses2 = ['TP_SITUACAO','IN_RESERVA_VAGAS','IN_FINANCIAMENTO_ESTUDANTIL','IN_APOIO_SOCIAL','IN_ATIVIDADE_EXTRACURRICULAR','TP_ESCOLA_CONCLUSAO_ENS_MEDIO','QT_CARGA_HORARIA_INTEG','NU_IDADE']
    p100matri= np.count_nonzero(busca.TP_SITUACAO==2,axis=0)/len(busca) *100
    p100tranca= np.count_nonzero(busca.TP_SITUACAO==3,axis=0)/len(busca) *100
    p100desv= np.count_nonzero(busca.TP_SITUACAO==4,axis=0)/len(busca) *100
    p100forma= np.count_nonzero(busca.TP_SITUACAO==6,axis=0)/len(busca) *100
    p100trans= np.count_nonzero(busca.TP_SITUACAO==5,axis=0)/len(busca) *100
#2015, 216
else:
    interesses1 = ['CO_ALUNO_SITUACAO','CO_CATEGORIA_ADMINISTRATIVA', 'CO_TURNO_ALUNO','CO_MODALIDADE_ENSINO','IN_SEXO_ALUNO','CO_NACIONALIDADE_ALUNO']
    interesses2 = ['CO_ALUNO_SITUACAO','IN_RESERVA_VAGAS','IN_FINANC_ESTUDANTIL','IN_APOIO_SOCIAL','IN_ATIVIDADE_EXTRACURRICULAR','CO_TIPO_ESCOLA_ENS_MEDIO','QT_CARGA_HORARIA_INTEG','NU_IDADE']
    p100matri= np.count_nonzero(busca.CO_ALUNO_SITUACAO==2,axis=0)/len(busca) *100
    p100tranca= np.count_nonzero(busca.CO_ALUNO_SITUACAO==3,axis=0)/len(busca) *100
    p100desv= np.count_nonzero(busca.CO_ALUNO_SITUACAO==4,axis=0)/len(busca) *100
    p100forma= np.count_nonzero(busca.CO_ALUNO_SITUACAO==6,axis=0)/len(busca) *100
    p100trans= np.count_nonzero(busca.CO_ALUNO_SITUACAO==5,axis=0)/len(busca) *100


#busca_temp = busca[busca.TP_SITUACAO.isin([2])]
cursos = busca.groupby('CO_CURSO')
stat = open((path+ano+'/stat.txt'),"w")
stat.write("Total_alunos:"+str(len(busca))+"\n")
stat.write("Total_cursos:"+str(len(cursos))+"\n")
stat.write('Matriculados:'+str(p100matri)+'\n')
stat.write('Trancamentos:'+str(p100tranca)+'\n')
stat.write('Desvinculados:'+str(p100desv)+'\n')
stat.write('Formados:'+str(p100forma)+'\n')
stat.write('Transferidos:'+str(p100trans)+'\n')
for (columnName, columnData) in busca.iteritems():
    stat.write(columnName+'\n')
    if(busca[columnName].dtype == 'O'):
        stat.write('Texto')
    else:
        stat.write('std:'+str(np.std(busca[columnName]))+"\n")
        stat.write('mean:'+str(np.mean(busca[columnName]))+"\n")
        stat.write('median:'+str(np.median(busca[columnName]))+"\n")
    stat.write(";\n")
stat.close()
busca_interesses1 = busca[interesses1]
if len(interesses1)==6:
    busca_interesses1['TP_DEFICIENCIA'] = [1 if (x ==1) else 0 for x in busca['IN_DEF_AUDITIVA']]
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_FISICA'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_INTELECTUAL'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_MULTIPLA'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_SURDEZ'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_SURDOCEGUEIRA'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_BAIXA_VISAO'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_CEGUEIRA'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
    busca_interesses1['TP_DEFICIENCIA'] = np.where(busca['IN_DEF_SUPERDOTACAO'] == 1, 1, busca_interesses1['TP_DEFICIENCIA'])
busca_interesses2 = busca[interesses2]


corr_geral= busca.corr(method='kendall')[interesses2[0]].sort_values(ascending=True)

for i in range(len(interesses1)-1):
    chisq_of_df_cols(busca_interesses1, interesses1[0], interesses1[i+1])        



corr_geral.to_csv(path+ano+'/correlacao.csv')
corr_rel= corr_geral.nsmallest(5).append(corr_geral.nlargest(6)[1:6])
plt.tight_layout()
corr_rel.plot.barh(rot=0).get_figure().savefig(path+ano+'/corr_geral.png',bbox_inches = "tight")
plt.clf()     # close the figure window
total = len(busca)

#plt.figure()
sns_plot = sns.pairplot(busca_interesses1)
sns_plot.savefig(path+ano+'/interesses1.png')
plt.clf()
#plt.figure()
sns_plot = sns.pairplot(busca_interesses2)
sns_plot.savefig(path+ano+'/interesses2.png')
plt.clf()
# Now if we normalize it by column:
df_norm_col=(busca_interesses1-busca_interesses1.mean())/busca_interesses1.std()


#plt.figure()

########################################################################################################################
situacao = ['Matriculados','Trancamentos','Desvinculados', 'Transferidos' ,'Formados']
cat_adm = ['Pública Federal','Pública Estadual','Pública Municipal', 'Privada lucrativa' ,'Privada n lucrativa','Especial']
if ano=='2016' or ano == '2015':   
    #generate_graph_percents(0,situacao,'dist_alunos',busca.CO_ALUNO_SITUACAO)
    generate_graph_percents(1,cat_adm,'matriculados',busca[busca.CO_ALUNO_SITUACAO ==2].CO_CATEGORIA_ADMINISTRATIVA)
    generate_graph_percents(1,cat_adm,'trancamentos',busca[busca.CO_ALUNO_SITUACAO ==3].CO_CATEGORIA_ADMINISTRATIVA)
    generate_graph_percents(1,cat_adm,'desvinculados',busca[busca.CO_ALUNO_SITUACAO ==4].CO_CATEGORIA_ADMINISTRATIVA)
    ax = sns.violinplot(x=busca.CO_ALUNO_SITUACAO,y=busca.QT_CARGA_HORARIA_INTEG,scale='width',dodge=False,cut=0)
    ax.set_title(ano)
    ax.set_xticklabels(['matriculas','trancamentos','desvinculados','transferidos','formados','falecidos'],rotation=45)
    ax.axes.set_ylim([0,12000])
    plt.savefig(path+ano+'/violin.png',bbox_inches = "tight")
else:
    #generate_graph_percents(0,situacao,'dist_alunos',busca.TP_SITUACAO)
    generate_graph_percents(1,cat_adm,'matriculados',busca[busca.TP_SITUACAO==2].TP_CATEGORIA_ADMINISTRATIVA)
    generate_graph_percents(1,cat_adm,'trancamentos',busca[busca.TP_SITUACAO==3].TP_CATEGORIA_ADMINISTRATIVA)
    generate_graph_percents(1,cat_adm,'desvinculados',busca[busca.TP_SITUACAO==4].TP_CATEGORIA_ADMINISTRATIVA)
    ax = sns.violinplot(x=busca.TP_SITUACAO,y=busca.QT_CARGA_HORARIA_INTEG,scale='width',dodge=False,cut=0)
    ax.set_title(ano)
    ax.set_xticklabels(['matriculas','trancamentos','desvinculados','transferidos','formados','falecidos'],rotation=45)
    ax.axes.set_ylim([0,12000])
    plt.savefig(path+ano+'/violin.png',bbox_inches = "tight")
generate_graph_avg('matriculados_qt',busca[busca.TP_SITUACAO==2].QT_CARGA_HORARIA_INTEG)
generate_graph_avg('trancamentos_qt',busca[busca.TP_SITUACAO==3].QT_CARGA_HORARIA_INTEG)
generate_graph_avg('desvinculados_qt',busca[busca.TP_SITUACAO==4].QT_CARGA_HORARIA_INTEG)
generate_graph_avg('formados_qt',busca[busca.TP_SITUACAO==6].QT_CARGA_HORARIA_INTEG)

#Courses
for name, group in cursos:
    generate_graph_percents(0,situacao,'cursos/'+str(name),group.TP_SITUACAO)
    
########################################################################################################################
#Cursos mais afetados
file1 = open((path+ano+'/cursos.txt'),"r")
df = pd.DataFrame(columns=['cod','matriculados','trancamentos','desvinculados','formados','transferidos'])
lines = file1.readlines()
count = 1
index = 0
df.loc[0]=[0,0,0,0,0,0]
for l in lines:
    if count==1:
        df.at[index,'cod']= int(l[6:])
    if count==2:
        df.at[index,'matriculados']= float(l[13:])
    if count==3:
        df.at[index,'trancamentos']= float(l[13:])
    if count==4:
        df.at[index,'desvinculados']= float(l[14:])
    if count==5:
        df.at[index,'formados']= float(l[9:])
    if count==6:
        df.at[index,'transferidos']= float(l[13:])
    count+=1
    if count==7:
        count=1
        index+=1
        df.loc[index]= [0,0,0,0,0,0]
df = df[:-1]
t = [x for x in range(10)]
t10trt=df.sort_values(by=['trancamentos'],ascending=False).head(10)
t10trt.plot.barh(x='cod',stacked=True,title=ano)
plt.savefig(path+ano+'/tranc.png',bbox_inches = "tight")
t10des=df.sort_values(by=['desvinculados'],ascending=False).head(10)
t10des.plot.barh(x='cod',stacked=True,title=ano)
plt.savefig(path+ano+'/desv.png',bbox_inches = "tight")
########################################################################################################################
#General data
busca = pd.read_csv(path+ano+'/geral_filtro.csv')
s=[]
if ano=='2017' or ano=='2018':
    s = pd.value_counts(busca['TP_SITUACAO'])
else:
    s = pd.value_counts(busca['CO_ALUNO_SITUACAO'])
x = ['Matriculados','Trancamentos','Desvinculados', 'Transferidos' ,'Formados','Falecidos']
s=s.sort_index(ascending=True)
s=s/s.sum() * 100
plt.barh(x,s.values)
plt.title(ano)
plt.savefig(path+ano+'/perc.png',bbox_inches = "tight")


########################################################################################################################
#Freq, similarities and clustering
busca = pd.read_csv(path+ano+'/geral_filtro.csv')

if ano=='2017' or ano=='2018':
    if(ano == '2018'):
        interesses1 = ['TP_SITUACAO','TP_CATEGORIA_ADMINISTRATIVA', 'TP_TURNO','TP_MODALIDADE_ENSINO','TP_SEXO','TP_NACIONALIDADE','IN_DEFICIENCIA']
    else:    
        interesses1 = ['TP_SITUACAO','TP_CATEGORIA_ADMINISTRATIVA', 'TP_TURNO','TP_MODALIDADE_ENSINO','TP_SEXO','TP_NACIONALIDADE','TP_DEFICIENCIA']
    interesses2 = ['IN_RESERVA_VAGAS','IN_FINANCIAMENTO_ESTUDANTIL','IN_APOIO_SOCIAL','IN_ATIVIDADE_EXTRACURRICULAR','TP_ESCOLA_CONCLUSAO_ENS_MEDIO','QT_CARGA_HORARIA_INTEG','NU_IDADE']
#2015, 216
else:
    interesses1 = ['CO_ALUNO_SITUACAO','CO_CATEGORIA_ADMINISTRATIVA', 'CO_TURNO_ALUNO','CO_MODALIDADE_ENSINO','IN_SEXO_ALUNO','CO_NACIONALIDADE_ALUNO']
    interesses2 = ['IN_RESERVA_VAGAS','IN_FINANC_ESTUDANTIL','IN_APOIO_SOCIAL','IN_ATIVIDADE_EXTRACURRICULAR','CO_TIPO_ESCOLA_ENS_MEDIO','QT_CARGA_HORARIA_INTEG','NU_IDADE_ALUNO']
busca = busca[interesses1+interesses2]
bsize = busca.columns.size
asso = pd.DataFrame(1, index=np.arange(bsize), columns=busca.columns)
for i in range(bsize):
    for j in range(i,bsize):
        asso.iloc[j,i]=cramers_v(busca.iloc[:,i],busca.iloc[:,j])
        asso.iloc[i,j]=cramers_v(busca.iloc[:,i],busca.iloc[:,j])
ax = plt.axes()
sns_plot=sns.heatmap(asso,yticklabels=busca.columns)
ax.set_title(ano)
sns_plot.figure.savefig(path+ano+'/correlations.png',bbox_inches = "tight")
plt.clf()

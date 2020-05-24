#########################################################################################################################
#Read pdf
#########################################################################################################################
from tabula import read_pdf
def read_MECCodigos(file):
    df= read_pdf(file,pages='all',lattice=True)
    file1 = open((file+'.txt'),"w")
    for j in range(1,len(df)):
        for  i in range(len(df[j])):
            file1.write(str(df[j]['Código\rCurso'][i]))
            file1.write("\n") 
    file1.close()        

read_MECCodigos("todos.pdf")
########################################################################################################################
#Create only bcc students
########################################################################################################################
# csv file name 
filename = "DM_ALUNO.CSV"
censo = pd.read_csv(filename,sep="|",encoding='ISO-8859-1')
print(censo.head())
print(censo.columns)
print(censo.shape)

#cursos
file_c = "todos.txt"
lista = []
booleans = [False] * len(censo)
i=0
with open(file_c) as f:
    alist = [line.rstrip() for line in f]
for i in alist:
    lista.append(i)
busca = censo[censo.CO_CURSO.isin(lista)]
busca.to_csv(r'geral_filtro.csv')
busca.CO_CURSO.nunique()
busca = pd.read_csv('C:/Users/morps/Desktop/dados/geral_filtro.csv').profile_report()
########################################################################################################################

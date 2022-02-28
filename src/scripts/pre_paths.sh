#Adjust to your own
if [ `echo ~` == '/home/garridoa' ]
then
  conda activate diora
  datapath=`echo  ~/bwfw/data/`
  srcpath=`echo ~/diora/pytorch/`
  modelspath=`echo ~/diora/log/`
  evalpath=`echo ~/bwfw/src/metrics/`
  bwfwsrcpath=`echo ~/bwfw/`
fi


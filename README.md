# Less bits is more
This is the PyTorch implementation of [Less bits is more: How pruning deep binary networks increases weight capacity](https://openreview.net/pdf?id=Hy8JM_Fvt5N)

## Illustrative 2D example
<table border=0 >
	<tbody>
    <tr>
			<td>  </td>
			<td align="center"> Full binary network:  combinations/solutions (512/30) </td>
			<td align="center"> Pruned subnetwork:  combinations/solutions (2304/109) </td>
			<td align="center"> Bi-half subnetwork: combinations/solutions (630/98) </td>
		</tr>
		<tr>
			<td width="19%" align="center"> Decision Boundaries </td>
			<td width="27%" > <img src="https://raw.githubusercontent.com/liyunqianggyn/Less-bits-is-more-How-pruning-deep-binary-networks-increases-weight-capacity/master/2DToyexample/fig/FullNet.png"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Less-bits-is-more-How-pruning-deep-binary-networks-increases-weight-capacity/master/2DToyexample/fig/Pruneoneweight.png"> </td>
			<td width="27%"> <img src="https://raw.githubusercontent.com/liyunqianggyn/Less-bits-is-more-How-pruning-deep-binary-networks-increases-weight-capacity/master/2DToyexample/fig/Pruneoneweight_half.png"> </td>
		</tr>
	</tbody>
</table>

## Architectures

   ### Shallow networks: Conv2/4/6/8

```
Shallow networks: Conv2, Conv4, Conv6 and Conv8
```


```
ImageNet
```

## Contact
If you have any problem about our code, feel free to contact

 - y.li-19@tudelft.nl


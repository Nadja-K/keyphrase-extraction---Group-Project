<template>
  <v-data-table
    v-model="selectedState"
    :headers="headers"
    :items="items"
    select-all
    item-key="name"
    class="fixed-header"
    hide-actions
  >
    <template slot="items" slot-scope="props">
      <tr :active="props.selected" @click="props.selected = !props.selected">
        <td>
          <v-checkbox
            :input-value="props.selected"
            primary
            hide-details
            :color="`#${palette[props.item.value]}`"
          ></v-checkbox>
        </td>
        <td v-for="key in Object.keys(props.item).filter(key => key !== 'value')"
            :key="key"
        >
            {{ props.item[key] }}
        </td>
      </tr>
    </template>
  </v-data-table>
</template>

<script>
import palette from '../palette';

export default {
    model: {
        prop: 'selected',
        event: 'change',
    },
    props: {
        headers: Array,
        items: Array,
        selected: Array,
    },
    computed: {
        selectedState: {
            get() {
                return this.selected;
            },
            set(value) {
                this.$emit('change', value);
            },
        },
    },
    data() {
        return {
            palette,
        };
    },
    methods: {
        toggleAll() {
            if (this.selected.length) this.selected = [];
            else this.selected = this.desserts.slice();
        },
    },
};
</script>

<style>
</style>
